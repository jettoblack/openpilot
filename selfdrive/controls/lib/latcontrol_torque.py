import math

from cereal import log
from common.numpy_fast import interp
from common.params import Params
from decimal import Decimal
from selfdrive.car.interfaces import CarInterfaceBase
from selfdrive.controls.lib.latcontrol import LatControl, MIN_STEER_SPEED
from selfdrive.controls.lib.pid import PIDController
from selfdrive.controls.lib.vehicle_model import ACCELERATION_DUE_TO_GRAVITY

# At higher speeds (25+mph) we can assume:
# Lateral acceleration achieved by a specific car correlates to
# torque applied to the steering rack. It does not correlate to
# wheel slip, or to speed.

# This controller applies torque to achieve desired lateral
# accelerations. To compensate for the low speed effects we
# use a LOW_SPEED_FACTOR in the error. Additionally, there is
# friction in the steering wheel that needs to be overcome to
# move it at all, this is compensated for too.

LOW_SPEED_X = [0, 10, 20, 30]
LOW_SPEED_Y = [225, 169, 100, 25]

def sign(x):
  return 1.0 if x > 0.0 else -1.0


class LatControlTorque(LatControl):
  def __init__(self, CP, CI):
    super().__init__(CP, CI)
    self.torque_params = CP.lateralTuning.torque
    self.pid = PIDController(self.torque_params.kp, self.torque_params.ki, k_d=self.torque_params.kd,
                             k_f=self.torque_params.kf, pos_limit=self.steer_max, neg_limit=-self.steer_max)
    self.torque_from_lateral_accel = CI.torque_from_lateral_accel()
    self.use_steering_angle = self.torque_params.useSteeringAngle
    self.steering_angle_deadzone_deg = self.torque_params.steeringAngleDeadzoneDeg

    self.params = Params()
    self._torque_max_lat_accel = 0
    self._torque_friction = 0
    self.frame = 0
    self.custom_torque = False
    self.custom_torque_timer = 0

  def update_live_torque_params(self, latAccelFactor, latAccelOffset, friction):
    self.torque_params.latAccelFactor = latAccelFactor
    self.torque_params.latAccelOffset = latAccelOffset
    self.torque_params.friction = friction

  def live_tune(self):
    self.frame += 1
    if self.frame % 300 == 0:
      self._torque_max_lat_accel = float(Decimal(self.params.get("TorqueMaxLatAccel", encoding="utf8")) * Decimal('0.01'))
      self._torque_friction = float(Decimal(self.params.get("TorqueFriction", encoding="utf8")) * Decimal('0.01'))
      self.torque_params.latAccelFactor = self._torque_max_lat_accel
      self.torque_params.friction = self._torque_friction
      self.frame = 0

  def update(self, active, CS, VM, params, last_actuators, steer_limited, desired_curvature, desired_curvature_rate, llk, mean_curvature=0.0):
    self.custom_torque_timer += 1
    if self.custom_torque_timer > 100:
      self.custom_torque_timer = 0
      self.custom_torque = self.params.get_bool("CustomTorqueLateral")
    if self.custom_torque:
      self.live_tune()
    pid_log = log.ControlsState.LateralTorqueState.new_message()

    if CS.vEgo < MIN_STEER_SPEED or not active:
      output_torque = 0.0
      pid_log.active = False
    else:
      if self.use_steering_angle:
        actual_curvature = -VM.calc_curvature(math.radians(CS.steeringAngleDeg - params.angleOffsetDeg), CS.vEgo, params.roll)
        curvature_deadzone = abs(VM.calc_curvature(math.radians(self.steering_angle_deadzone_deg), CS.vEgo, 0.0))
      else:
        actual_curvature_vm = -VM.calc_curvature(math.radians(CS.steeringAngleDeg - params.angleOffsetDeg), CS.vEgo, params.roll)
        actual_curvature_llk = llk.angularVelocityCalibrated.value[2] / CS.vEgo
        actual_curvature = interp(CS.vEgo, [2.0, 5.0], [actual_curvature_vm, actual_curvature_llk])
        curvature_deadzone = 0.0
      desired_lateral_accel = desired_curvature * CS.vEgo ** 2
      desired_lateral_jerk = desired_curvature_rate * CS.vEgo**2

      # desired rate is the desired rate of change in the setpoint, not the absolute desired curvature
      # desired_lateral_jerk = desired_curvature_rate * CS.vEgo ** 2
      actual_lateral_accel = actual_curvature * CS.vEgo ** 2
      lateral_accel_deadzone = curvature_deadzone * CS.vEgo ** 2

      low_speed_factor = interp(CS.vEgo, LOW_SPEED_X, LOW_SPEED_Y)
      setpoint = desired_lateral_accel + low_speed_factor * min(abs(desired_curvature), abs(mean_curvature)) * sign(desired_curvature)
      measurement = actual_lateral_accel + low_speed_factor * min(abs(actual_curvature), abs(mean_curvature)) * sign(actual_curvature)
      error = setpoint - measurement
      gravity_lateral_accel = -params.roll * ACCELERATION_DUE_TO_GRAVITY
      pid_log.error = CarInterfaceBase.torque_from_lateral_accel_linear(error, self.torque_params, error,
                                                     lateral_accel_deadzone, friction_compensation=False,
                                                      v_ego=CS.vEgo, g_lat_accel=0., lateral_jerk_desired=0.0)
      
      ff = self.torque_from_lateral_accel(desired_lateral_accel, self.torque_params,
                                          desired_lateral_accel - actual_lateral_accel,
                                          lateral_accel_deadzone, friction_compensation=True,
                                          v_ego=CS.vEgo, g_lat_accel=gravity_lateral_accel, lateral_jerk_desired=desired_lateral_jerk)

      freeze_integrator = steer_limited or CS.steeringPressed or CS.vEgo < 5
      output_torque = self.pid.update(pid_log.error,
                                      feedforward=ff,
                                      speed=CS.vEgo,
                                      freeze_integrator=freeze_integrator)

      pid_log.active = True
      pid_log.p = self.pid.p
      pid_log.i = self.pid.i
      pid_log.d = self.pid.d
      pid_log.f = self.pid.f
      pid_log.output = -output_torque
      pid_log.actualLateralAccel = actual_lateral_accel
      pid_log.desiredLateralAccel = desired_lateral_accel
      pid_log.saturated = self._check_saturation(self.steer_max - abs(output_torque) < 1e-3, CS, steer_limited)

    # TODO left is positive in this convention
    return -output_torque, 0.0, pid_log
