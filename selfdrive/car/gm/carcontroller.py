from cereal import car
from common.conversions import Conversions as CV
from common.filter_simple import FirstOrderFilter
from common.numpy_fast import clip, interp
from common.realtime import DT_CTRL, sec_since_boot
import math
from opendbc.can.packer import CANPacker
from selfdrive.car import apply_std_steer_torque_limits
from selfdrive.car.gm import gmcan
from selfdrive.car.gm.values import DBC, CanBus, CarControllerParams, CruiseButtons, EV_CAR
from selfdrive.controls.lib.drive_helpers import apply_deadzone
from selfdrive.controls.lib.pid import PIDController
from selfdrive.controls.lib.vehicle_model import ACCELERATION_DUE_TO_GRAVITY

VisualAlert = car.CarControl.HUDControl.VisualAlert
NetworkLocation = car.CarParams.NetworkLocation
LongCtrlState = car.CarControl.Actuators.LongControlState

# Camera cancels up to 0.1s after brake is pressed, ECM allows 0.5s
CAMERA_CANCEL_DELAY_FRAMES = 10

BRAKE_PITCH_FACTOR_BP = [5., 10.] # [m/s] smoothly revert to planned accel at low speeds
BRAKE_PITCH_FACTOR_V = [0., 1.] # [unitless in [0,1]; don't touch]
PITCH_DEADZONE = 0.01 # [radians] 0.01 ≈ 1% grade
PITCH_MAX_DELTA = math.radians(10.0) * DT_CTRL * 4  # 10°/s, checked at 25Hz
PITCH_MIN, PITCH_MAX = math.radians(-19), math.radians(19) # steepest roads in US are ~18°

ONE_PEDAL_ACCEL_PITCH_FACTOR_BP = [4., 8.] # [m/s]
ONE_PEDAL_ACCEL_PITCH_FACTOR_V = [0.4, 1.] # [unitless in [0-1]]
ONE_PEDAL_ACCEL_PITCH_FACTOR_INCLINE_V = [0.2, 1.] # [unitless in [0-1]]

ONE_PEDAL_MODE_DECEL_BP = [i * CV.MPH_TO_MS for i in [0.5, 6.]] # [mph to meters]
ONE_PEDAL_MODE_DECEL_V = [-1.0, -1.1]
ONE_PEDAL_MIN_SPEED = 2.1
ONE_PEDAL_DECEL_RATE_LIMIT_UP = 0.8 * DT_CTRL * 4 # m/s^2 per second for increasing braking force
ONE_PEDAL_DECEL_RATE_LIMIT_DOWN = 0.8 * DT_CTRL * 4 # m/s^2 per second for decreasing

ONE_PEDAL_MAX_DECEL = -3.5
ONE_PEDAL_SPEED_ERROR_FACTOR_BP = [1.5, 20.] # [m/s] 
ONE_PEDAL_SPEED_ERROR_FACTOR_V = [0.4, 0.2] # factor of error for non-lead braking decel

ONE_PEDAL_LEAD_ACCEL_RATE_LOCKOUT_T = 0.6 # [s]

class CarController:
  def __init__(self, dbc_name, CP, VM):
    self.CP = CP
    self.start_time = 0.
    self.apply_steer_last = 0
    self.apply_gas = 0
    self.apply_brake = 0
    self.frame = 0
    self.last_steer_frame = 0
    self.last_button_frame = 0
    self.cancel_counter = 0

    self.lka_steering_cmd_counter = 0
    self.sent_lka_steering_cmd = False
    self.lka_icon_status_last = (False, False)

    self.params = CarControllerParams(self.CP)
    self.pitch = FirstOrderFilter(0., 0.09 * 4, DT_CTRL * 4) # runs at 25 Hz
    
        # pid runs at 25Hz
    self.one_pedal_pid = PIDController(k_p=(CP.longitudinalTuning.kpBP, CP.longitudinalTuning.kpV), 
                                      k_i=(CP.longitudinalTuning.kiBP, CP.longitudinalTuning.kiV), 
                                      k_d=(CP.longitudinalTuning.kdBP, CP.longitudinalTuning.kdV),
                                      derivative_period=0.1, neg_limit=ONE_PEDAL_MAX_DECEL, pos_limit=0.0,
                                      rate=1/(DT_CTRL * 4))
    self.one_pedal_decel = 0.0
    self.one_pedal_decel_in = 0.
    self.lead_accel_last_t = 0.
    self.one_pedal_mode_op_braking_allowed = True

    self.packer_pt = CANPacker(DBC[self.CP.carFingerprint]['pt'])
    self.packer_obj = CANPacker(DBC[self.CP.carFingerprint]['radar'])
    self.packer_ch = CANPacker(DBC[self.CP.carFingerprint]['chassis'])

  def update(self, CC, CS):
    actuators = CC.actuators
    accel = actuators.accel
    hud_control = CC.hudControl
    hud_alert = hud_control.visualAlert
    hud_v_cruise = hud_control.setSpeed
    if hud_v_cruise > 70:
      hud_v_cruise = 0

    # Send CAN commands.
    can_sends = []

    # Steering (Active: 50Hz, inactive: 10Hz)
    # Attempt to sync with camera on startup at 50Hz, first few msgs are blocked
    init_lka_counter = not self.sent_lka_steering_cmd and self.CP.networkLocation == NetworkLocation.fwdCamera
    steer_step = self.params.INACTIVE_STEER_STEP
    if CC.latActive or init_lka_counter:
      steer_step = self.params.ACTIVE_STEER_STEP

    # Avoid GM EPS faults when transmitting messages too close together: skip this transmit if we just received the
    # next Panda loopback confirmation in the current CS frame.
    if CS.loopback_lka_steering_cmd_updated:
      self.lka_steering_cmd_counter += 1
      self.sent_lka_steering_cmd = True
    elif (self.frame - self.last_steer_frame) >= steer_step:
      # Initialize ASCMLKASteeringCmd counter using the camera until we get a msg on the bus
      if init_lka_counter:
        self.lka_steering_cmd_counter = CS.camera_lka_steering_cmd_counter + 1

      if CC.latActive:
        new_steer = int(round(actuators.steer * self.params.STEER_MAX))
        apply_steer = apply_std_steer_torque_limits(new_steer, self.apply_steer_last, CS.out.steeringTorque, self.params)
      else:
        apply_steer = 0

      self.last_steer_frame = self.frame
      self.apply_steer_last = apply_steer
      idx = self.lka_steering_cmd_counter % 4
      can_sends.append(gmcan.create_steering_control(self.packer_pt, CanBus.POWERTRAIN, apply_steer, idx, CC.latActive))



    
    if self.CP.openpilotLongitudinalControl:
      # Gas/regen, brakes, and UI commands - all at 25Hz
      if self.frame % 4 == 0:
        # Pitch compensated acceleration;
        # TODO: include future pitch (sm['modelDataV2'].orientation.y) to account for long actuator delay
        pitch = clip(CC.orientationNED[1], self.pitch.x - PITCH_MAX_DELTA, self.pitch.x + PITCH_MAX_DELTA)
        pitch = clip(pitch, PITCH_MIN, PITCH_MAX)
        self.pitch.update(pitch)
        accel_g = ACCELERATION_DUE_TO_GRAVITY * apply_deadzone(self.pitch.x, PITCH_DEADZONE) # driving uphill is positive pitch
        accel += accel_g
        
        
        if self.CP.carFingerprint in EV_CAR:
          if not CC.longActive:
            one_pedal_speed = max(CS.out.vEgo, ONE_PEDAL_MIN_SPEED)
          else:
            one_pedal_speed = CS.out.vEgo
          threshold_accel = self.params.update_ev_gas_brake_threshold(one_pedal_speed)
        else:
          threshold_accel = CS.out.aEgo
        
        if not CC.longActive:
          # ASCM sends max regen when not enabled
          self.apply_gas = self.params.INACTIVE_REGEN
          self.apply_brake = 0
        else:
          brake_accel = actuators.accel + accel_g * interp(CS.out.vEgo, BRAKE_PITCH_FACTOR_BP, BRAKE_PITCH_FACTOR_V)
          if self.CP.carFingerprint in EV_CAR:
            self.params.update_ev_gas_brake_threshold(CS.out.vEgo)
            self.apply_gas = int(round(interp(accel, self.params.EV_GAS_LOOKUP_BP, self.params.GAS_LOOKUP_V)))
            self.apply_brake = int(round(interp(brake_accel, self.params.EV_BRAKE_LOOKUP_BP, self.params.BRAKE_LOOKUP_V)))
          else:
            self.apply_gas = int(round(interp(accel, self.params.GAS_LOOKUP_BP, self.params.GAS_LOOKUP_V)))
            self.apply_brake = int(round(interp(brake_accel, self.params.BRAKE_LOOKUP_BP, self.params.BRAKE_LOOKUP_V)))

        idx = (self.frame // 4) % 4

        if CS.out.cruiseState.available and not CC.longActive and CS.auto_hold and CS.autohold_active and not CS.out.gasPressed and CS.out.gearShifter in ['drive','low'] and CS.out.vEgo < 0.02 and not CS.regen_paddle_pressed:
          # Auto Hold State
          car_stopping = self.apply_gas < self.params.ZERO_GAS
          at_full_stop = CS.out.standstill and car_stopping
          friction_brake_bus = CanBus.CHASSIS
          # GM Camera exceptions
          # TODO: can we always check the longControlState?
          if self.CP.networkLocation == NetworkLocation.fwdCamera:
            friction_brake_bus = CanBus.POWERTRAIN
          near_stop = (CS.out.vEgo < self.params.NEAR_STOP_BRAKE_PHASE) and car_stopping
          can_sends.append(gmcan.create_friction_brake_command(self.packer_ch, friction_brake_bus, self.apply_brake, idx, CC.enabled, near_stop, at_full_stop, self.CP))
          can_sends.append(gmcan.create_gas_regen_command(self.packer_pt, CanBus.POWERTRAIN, self.apply_gas, idx, CC.enabled and CS.out.cruiseState.enabled, at_full_stop))
          CS.autohold_activated = True
          self.one_pedal_pid.reset()
        elif CS.one_pedal_mode_active and CS.out.cruiseState.available and CS.out.gearShifter in ['drive','low'] and not (CC.longActive or CS.out.gasPressed or CS.out.brakePressed):
          t = sec_since_boot()
          self.one_pedal_decel_in = interp(CS.out.vEgo, ONE_PEDAL_MODE_DECEL_BP, ONE_PEDAL_MODE_DECEL_V)
          if self.one_pedal_mode_op_braking_allowed:
            self.one_pedal_decel_in = min(self.one_pedal_decel_in, CS.lead_accel)
            if CS.lead_accel <= self.one_pedal_decel_in:
              self.lead_accel_last_t = t
          
          if not self.one_pedal_mode_op_braking_allowed or CS.lead_accel != self.one_pedal_decel_in:
            error_factor = interp(CS.out.vEgo, ONE_PEDAL_SPEED_ERROR_FACTOR_BP, ONE_PEDAL_SPEED_ERROR_FACTOR_V)
          else:
            error_factor = 1.0
          error = self.one_pedal_decel_in - min(0.0, CS.out.aEgo + accel_g)
          error *= error_factor
          one_pedal_decel = self.one_pedal_pid.update(error, speed=CS.out.vEgo, feedforward=self.one_pedal_decel_in)
          if t - self.lead_accel_last_t > ONE_PEDAL_LEAD_ACCEL_RATE_LOCKOUT_T:
            self.one_pedal_decel = clip(one_pedal_decel, self.one_pedal_decel - ONE_PEDAL_DECEL_RATE_LIMIT_UP, self.one_pedal_decel + ONE_PEDAL_DECEL_RATE_LIMIT_DOWN)
          else:
            self.one_pedal_decel = one_pedal_decel
          self.one_pedal_decel = max(self.one_pedal_decel, ONE_PEDAL_MAX_DECEL)
          one_pedal_apply_brake = interp(self.one_pedal_decel, self.params.BRAKE_LOOKUP_BP, self.params.BRAKE_LOOKUP_V)
          self.apply_brake = int(round(one_pedal_apply_brake))
          at_full_stop = CS.out.standstill
          near_stop = CS.out.vEgo < self.params.NEAR_STOP_BRAKE_PHASE
          friction_brake_bus = CanBus.CHASSIS
          # GM Camera exceptions
          if self.CP.networkLocation == NetworkLocation.fwdCamera:
            friction_brake_bus = CanBus.POWERTRAIN
          can_sends.append(gmcan.create_friction_brake_command(self.packer_ch, friction_brake_bus, self.apply_brake, idx, CC.enabled, near_stop, at_full_stop, self.CP))
          can_sends.append(gmcan.create_gas_regen_command(self.packer_pt, CanBus.POWERTRAIN, self.apply_gas, idx, CC.enabled and CS.out.cruiseState.enabled, at_full_stop))
          CS.autohold_activated = False
        else:  
          if CS.out.gasPressed:
            at_full_stop = False
            near_stop = False
            car_stopping = False
          else:
            at_full_stop = CC.longActive and CS.out.standstill
            near_stop = CC.longActive and (CS.out.vEgo < self.params.NEAR_STOP_BRAKE_PHASE)
          friction_brake_bus = CanBus.CHASSIS
          # GM Camera exceptions
          # TODO: can we always check the longControlState?
          if self.CP.networkLocation == NetworkLocation.fwdCamera:
            at_full_stop = at_full_stop and actuators.longControlState == LongCtrlState.stopping
            friction_brake_bus = CanBus.POWERTRAIN

          # GasRegenCmdActive needs to be 1 to avoid cruise faults. It describes the ACC state, not actuation
          can_sends.append(gmcan.create_gas_regen_command(self.packer_pt, CanBus.POWERTRAIN, self.apply_gas, idx, CC.enabled and CS.out.cruiseState.enabled, at_full_stop))
          can_sends.append(gmcan.create_friction_brake_command(self.packer_ch, friction_brake_bus, self.apply_brake, idx, CC.enabled, near_stop, at_full_stop, self.CP))
          CS.autohold_activated = False
          self.one_pedal_pid.reset()

        # Send dashboard UI commands (ACC status)
        send_fcw = hud_alert == VisualAlert.fcw
        can_sends.append(gmcan.create_acc_dashboard_command(self.packer_pt, CanBus.POWERTRAIN, CC.enabled and CS.out.cruiseState.enabled,
                                                            hud_v_cruise * CV.MS_TO_KPH, hud_control.leadVisible, send_fcw))
      else:
        accel_g = ACCELERATION_DUE_TO_GRAVITY * apply_deadzone(self.pitch.x, PITCH_DEADZONE) # driving uphill is positive pitch
        accel += accel_g

      # Radar needs to know current speed and yaw rate (50hz),
      # and that ADAS is alive (10hz)
      if not self.CP.radarOffCan:
        tt = self.frame * DT_CTRL
        time_and_headlights_step = 10
        if self.frame % time_and_headlights_step == 0:
          idx = (self.frame // time_and_headlights_step) % 4
          can_sends.append(gmcan.create_adas_time_status(CanBus.OBSTACLE, int((tt - self.start_time) * 60), idx))
          can_sends.append(gmcan.create_adas_headlights_status(self.packer_obj, CanBus.OBSTACLE))

        speed_and_accelerometer_step = 2
        if self.frame % speed_and_accelerometer_step == 0:
          idx = (self.frame // speed_and_accelerometer_step) % 4
          can_sends.append(gmcan.create_adas_steering_status(CanBus.OBSTACLE, idx))
          can_sends.append(gmcan.create_adas_accelerometer_speed_status(CanBus.OBSTACLE, CS.out.vEgo, idx))

      if self.CP.networkLocation == NetworkLocation.gateway and self.frame % self.params.ADAS_KEEPALIVE_STEP == 0:
        can_sends += gmcan.create_adas_keepalive(CanBus.POWERTRAIN)

    else:
      # While car is braking, cancel button causes ECM to enter a soft disable state with a fault status.
      # A delayed cancellation allows camera to cancel and avoids a fault when user depresses brake quickly
      self.cancel_counter = self.cancel_counter + 1 if CC.cruiseControl.cancel else 0

      # Stock longitudinal, integrated at camera
      if (self.frame - self.last_button_frame) * DT_CTRL > 0.04:
        if self.cancel_counter > CAMERA_CANCEL_DELAY_FRAMES:
          self.last_button_frame = self.frame
          can_sends.append(gmcan.create_buttons(self.packer_pt, CanBus.CAMERA, CS.buttons_counter, CruiseButtons.CANCEL))

    if self.CP.networkLocation == NetworkLocation.fwdCamera:
      # Silence "Take Steering" alert sent by camera, forward PSCMStatus with HandsOffSWlDetectionStatus=1
      if self.frame % 10 == 0:
        can_sends.append(gmcan.create_pscm_status(self.packer_pt, CanBus.CAMERA, CS.pscm_status))

    # Show green icon when LKA torque is applied, and
    # alarming orange icon when approaching torque limit.
    # If not sent again, LKA icon disappears in about 5 seconds.
    # Conveniently, sending camera message periodically also works as a keepalive.
    lka_active = CS.lkas_status == 1
    lka_critical = lka_active and abs(actuators.steer) > 0.9
    lka_icon_status = (lka_active, lka_critical)

    # SW_GMLAN not yet on cam harness, no HUD alerts
    if self.CP.networkLocation != NetworkLocation.fwdCamera and (self.frame % self.params.CAMERA_KEEPALIVE_STEP == 0 or lka_icon_status != self.lka_icon_status_last):
      steer_alert = hud_alert in (VisualAlert.steerRequired, VisualAlert.ldw)
      can_sends.append(gmcan.create_lka_icon_command(CanBus.SW_GMLAN, lka_active, lka_critical, steer_alert))
      self.lka_icon_status_last = lka_icon_status

    new_actuators = actuators.copy()
    new_actuators.accel = accel
    new_actuators.steer = self.apply_steer_last / self.params.STEER_MAX
    new_actuators.gas = self.apply_gas
    new_actuators.brake = self.apply_brake

    self.frame += 1
    return new_actuators, can_sends
