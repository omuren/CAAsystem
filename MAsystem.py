import cutils
import numpy as np
from scipy.optimize import minimize
from mpc_utils import model, generate_ref, simp_model    ###mpc_utilsに自動車のモデルが入ってる
#from trusty import init_predictor
import timeit
import G29ConnectorGateway as g29 #追加

# state variables order
# x
# y
# th
# v
# cte
# eth

# input variables order
# steer angle
# acceleration

init_state = np.array([0.0] * 6)
class Controller(object):
    def __init__(self, start_point, end_point, P =80, R = 2.0,): #P=6 ホライズン, rho: 人の限界は0.04ちょいくらい
        self.vars                = cutils.CUtils()
        self._current_x          = 0
        self._current_y          = 0
        self._current_yaw        = 0
        self._current_speed      = 0
        self._desired_speed      = 0
        self._vel_x              = 0
        self._vel_y              = 0
        self._current_timestamp  = 0
        self._start_control_loop = False
        self._set_throttle       = 0
        self._set_brake          = 0
        self._current_frame      = 0
        self._set_steer          = 0
        self._conv_rad_to_steer  = 180.0 / 70.0 / np.pi
        self._pi                 = np.pi
        self._2pi                = 2.0 * np.pi
        self.STEP_TIME           = 0.01
        self.R                   = R
        self.beta                =0
        self.alpha               =0
        self.P                   = P
        self.trusts              = []
        self.obs_pos             = np.array([])
        self.obs_vel             = np.array([])
        self.snaptime            = 0
        self.update_waypoints(start_point, end_point, 0.0)
        #self.trusty = init_predictor()
        #self.trusty.predict(["resources/init_pic.png"])

    def update_values(self, x, y, yaw, speed, timestamp, vel, frame, snaptime):
        self._current_x         = x
        self._current_y         = y
        self._current_yaw       = yaw
        self._current_speed     = speed
        self._current_timestamp = timestamp
        self._vel_x             = vel.x
        self._vel_y             = vel.y
        self._current_frame     = frame
        self.snaptime           = snaptime
        if self._current_frame:
            self._start_control_loop = True

    def update_desired_speed(self):
        min_idx       = 0
        min_dist      = float("inf")
        desired_speed = 0
        for i in range(len(self._waypoints)):
            dist = np.linalg.norm(np.array([
                    self._waypoints[i][0] - self._current_x,
                    self._waypoints[i][1] - self._current_y]))
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        if min_idx < len(self._waypoints)-1:
            desired_speed = self._waypoints[min_idx][2]
        else:
            desired_speed = self._waypoints[-1][2]
        self._desired_speed = desired_speed

    def update_waypoints(self, x_c, x_g, current_speed):
        self._waypoints = generate_ref(x_c, x_g, 20, current_speed)

    @staticmethod
    def calculate_gamma(trust, gamma_init = 0.08, delta = 0.55, lambdaa = 2):
        '''Converts trust values to gamma for cbf constraints
        '''
        return gamma_init + delta*(trust**lambdaa)

    def perceive(self, obs_pos, obs_vel):
        self.obs_pos = np.array(sorted(obs_pos, key = lambda x: x[0]))
        self.obs_vel = np.array(sorted(obs_vel, key = lambda x: x[0]))

    def get_commands(self):
        return self._set_throttle, self._set_steer, self._set_brake, self.alpha, self.beta

    def set_throttle(self, input_throttle):
        # Clamp the throttle command to valid bounds
        throttle           = np.fmax(np.fmin(input_throttle, 1.0), 0.0)
        self._set_throttle = throttle

    def set_steer(self, input_steer_in_rad):
        # Covnert radians to [-1, 1]
        input_steer = self._conv_rad_to_steer * input_steer_in_rad

        # Clamp the steering command to valid bounds
        steer           = np.fmax(np.fmin(input_steer, 1.0), -1.0)
        self._set_steer = steer

    def set_brake(self, input_brake):
        # Clamp the steering command to valid bounds
        brake           = np.fmax(np.fmin(input_brake, 1.0), 0.0)
        self._set_brake = brake

    def map_coord_2_Car_coord(self, x, y, yaw, waypoints):

        wps = np.squeeze(waypoints)
        wps_x = wps[:,0]
        wps_y = wps[:,1]

        num_wp = wps.shape[0]

        ## create the Matrix with 3 vectors for the waypoint x and y coordinates w.r.t. car
        wp_vehRef = np.zeros(shape=(3, num_wp))
        cos_yaw = np.cos(-yaw)
        sin_yaw = np.sin(-yaw)

        wp_vehRef[0,:] = cos_yaw * (wps_x - x) - sin_yaw * (wps_y - y)
        wp_vehRef[1,:] = sin_yaw * (wps_x - x) + cos_yaw * (wps_y - y)

        return wp_vehRef
                                                                                ###ここから制御器の設計　最適化問題をといてuを決定している
    def update_controls(self):                                                      ##ここの最適化問題消して自分用にしてみる
        x               = self._current_x
        y               = self._current_y
        yaw             = self._current_yaw
        v               = self._current_speed
        self.update_desired_speed()
        v_desired       = self._desired_speed
        t               = self._current_timestamp
        waypoints       = self._waypoints


        self.vars.create_var('x_previous', 0.0)
        self.vars.create_var('y_previous', 0.0)
        self.vars.create_var('th_previous', 0.0)
        self.vars.create_var('v_previous', 0.0)
        self.vars.create_var('cte_previous', 0.0)
        self.vars.create_var('eth_previous', 0.0)
        self.vars.create_var('t_previous', 0.0)
        self.vars.create_var('snaptime', 0.0)
        self.vars.create_var('prev_input', np.array([0,0.4]))

        ## step time ##
        self.STEP_TIME = t - self.vars.t_previous
        #print(self.STEP_TIME)

        ## init geuss ##
        u0 = self.vars.prev_input

        wps_vehRef = self.map_coord_2_Car_coord(x, y, yaw, waypoints)
        wps_vehRef_x = wps_vehRef[0,:]
        wps_vehRef_y = wps_vehRef[1,:]

        ## find COFF of the polynomial ##
        coff = np.polyfit(wps_vehRef_x, wps_vehRef_y, 3)

        if self._start_control_loop:
            def reference(X_obs, uh, u0, u_MA,r, beta=0.997, rho = 0.5):
                init_state = np.array([x,y,yaw,v])
                init_state_1 = init_state
                u_ref = uh
                r2 = r**2
                ip = abs(u0[0] - self._conv_rad_to_steer * uh[0])

                for i in range(self.P):
                    #X_obs = ##????????
                    u = beta * uh + (1 - beta) * u_MA
                    next_state = simp_model(u, init_state_1, dt = 0.014, L = 3) #dt = self.STEP_TIM

                    next_pos = np.array([next_state[0], next_state[1]])
                    distance = np.sum((next_pos - X_obs) ** 2)
                    if distance <= r2:
                        u_ref = u_MA
                        self.beta = beta
                        self.alpha = 1
                        break

                    init_state_1 = next_state
                    uh = u

                if distance > r2:
                    self.alpha = 0
                    if ip < rho:
                        self.beta = 0
                return u_ref

            def uma(X_obs, x, y, yaw):
                obs = -(X_obs[0]-x)*np.sin(yaw) + (X_obs[1]-y)*np.cos(yaw)
                if obs > 0:
                    u_MA0 = -1.22
                else:
                    u_MA0 = 1.22
                return u_MA0

            def obstacle(obs_pos, x, y):
                length = len(obs_pos)-1
                r = np.sum((np.array([x, y]) - obs_pos[0, :]) ** 2)
                X_obs = obs_pos[0, :]
                for i in range(length):
                    r_new = np.sum((np.array([x,y]) -obs_pos[i+1,:]) ** 2)
                    if r_new < r:
                        r = r_new
                        X_obs = obs_pos[i+1, :]
                return X_obs




            #とりあえず
            steer_output = g29.scalars()["steering_rad"] * 0.05
            brake_output = 0
            throttle_output = 0.48

            #テスト用
            if v < 7:
                throttle_output = 1.0

            if len(self.obs_pos) >= 1:
                uh = np.array([steer_output,throttle_output])  #ここをはんどるの入力にする
                r = self.R
                X_obs = obstacle(self.obs_pos,x,y)              #障害物と自動車の距離が一番近いものにしたい
                u_MA = np.array([uma(X_obs, x, y, yaw), uh[1]])
                u_ref = reference(X_obs, uh, u0, u_MA, r)
                beta = self.beta
                u = beta*u0+(1-beta)*u_ref
                self.vars.prev_input = u
                steer_output = u[0]         ##ステア角のoutputをしている．ここを(1-alpha)u_h+alpha u_aにすればよさそう
                throttle_output = u[1]


            self.set_throttle(throttle_output)  # in percent (0 to 1)
            self.set_steer(steer_output)        # in rad (-1.22 to 1.22)
            self.set_brake(brake_output)        # in percent (0 to 1)
            #print(self.beta)

        self.vars.t_previous = t  # Store timestamp  to be used in next step
        self.vars.v_previous = v
