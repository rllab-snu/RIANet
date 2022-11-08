from collections import deque
import numpy as np

class PID_controller(object):
    def __init__(self, direc_p=0.8, direc_i=0.5, direc_d=0.3, \
                        speed_p=5.0, speed_i=0.5, speed_d=1.0, win_size=40):
        
        self.direc_p, self.direc_i, self.direc_d = direc_p, direc_i, direc_d
        self.speed_p, self.speed_i, self.speed_d = speed_p, speed_i, speed_d        
        self.adjust_coeff = 0.1
        self.win_size=40

        self.direc_error_window = deque(np.zeros(win_size), maxlen=win_size)
        self.speed_error_window = deque(np.zeros(win_size), maxlen=win_size)

    def reset(self):                
        del self.direc_error_window 
        del self.speed_error_window

        self.direc_error_window = deque(np.zeros(self.win_size), maxlen=self.win_size)
        self.speed_error_window = deque(np.zeros(self.win_size), maxlen=self.win_size)

    def get_error(self, pos, compass, speed, vehicle_st, target_pos, target_speed):
        # pos, target_pos : np.array size 2, others: float 
        R = np.array([np.cos(compass), -np.sin(compass), np.sin(compass), np.cos(compass)]).reshape(2,2)
        direc_vec = np.matmul((target_pos - pos).reshape(1,2), R)
        
        direc_error = -np.arctan2(direc_vec[:,1], direc_vec[:,0]).item()

        pos_error_adjustment = vehicle_st.squeeze()[1].item()
        direc_error += self.adjust_coeff * pos_error_adjustment

        speed_error = target_speed - speed         

        return direc_error, speed_error

    def control(self, pos, compass, speed, vehicle_st, target_pos, target_speed, steer_adjustment=0):
        # pos, target_pos : np.array size 2, others: float 
        direc_error, speed_error = self.get_error(pos, compass, speed, vehicle_st, target_pos, target_speed)
        direc_error += self.adjust_coeff * (-steer_adjustment)

        direc_error = np.clip(direc_error, -1, 1)
        speed_error = np.clip(speed_error, 0, 0.25)

        self.direc_error_window.append(direc_error)
        self.speed_error_window.append(speed_error)

        steer = self.direc_p * direc_error + \
                self.direc_i * np.mean(self.direc_error_window) + \
                self.direc_d * (self.direc_error_window[-1] - self.direc_error_window[-2])

        steer = float(np.clip(steer, -1.0, 1.0))

        throttle = self.speed_p * speed_error + \
                   self.speed_i * np.mean(self.speed_error_window) + \
                   self.speed_d * (self.speed_error_window[-1] - self.speed_error_window[-2])

        throttle = float(np.clip(throttle, 0.0, 0.75))

        #print(round(direc_error,4), round(vehicle_st.squeeze()[1].item(),4), round(steer,4), round(throttle,4)) 

        return steer, throttle
