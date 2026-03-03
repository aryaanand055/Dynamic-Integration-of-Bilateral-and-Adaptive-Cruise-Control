'''
City .py: Contains the City class for managing the traffic simulation.
'''

from car import Car
from road import Road
import numpy as np
import math as math

class City:
    def __init__(self):
        self.cars = []
        self.roads = []
        self.step_count = 0
        self.model = 'ACC'  
        self.mode = 'BCC'
        self.min_gap = 5.0 
        self.dt = 0.1 
        self.overall_min_gap = float('inf')
        self.overall_max_gap = 0
        self.all_gaps = []

    def init(self, car_number, kd, kv, kc, v_des, max_v, min_v, min_dis, reaction_time, headway_time, max_a, min_a, min_gap=5.0, dt=0.1, model='ACC'):
        # Reset simulation state
        self.cars.clear()
        self.roads.clear()
        self.step_count = 0
        self.model = model
        self.dt = dt
        self.overall_min_gap = float('inf')
        self.overall_max_gap = 0
        self.all_gaps = []

        # Create a single straight road for simplicity
        road = Road(1000, 0, 0, 1, 0)
        self.roads.append(road)

        # Place cars at intervals along the road
        for i in range(int(car_number)):
            # Initial velocity, position, and sizeof each car
            velocity = v_des
            car_length = 4
            headway = min_dis + velocity * reaction_time
            pos = 1000 - (car_number - 1 -i) * (car_length + headway) 
            if i == 0:
                color = 'red'  
            elif i == car_number - 1:
                color = 'green'
            else:
                color = 'blue'
            car = Car(length=car_length, color=color, pos=pos,min_dis=min_dis, velocity=velocity, acceleration=0, current_road=road)
            car.original_color = color  
            car.collision_timer = 0
            self.cars.append(car)
            road.enter_road(car)

        # Store model parameters
        self.kd = kd
        self.kv = kv
        self.kc = kc
        self.v_des = v_des
        self.initial_v_des = v_des
        self.max_v = max_v
        self.min_v = min_v
        self.reaction_time = reaction_time
        self.headway_time = headway_time
        self.max_a = max_a
        self.min_a = min_a
        self.min_dis = min_dis
        self.min_gap = min_gap

    def run(self, dt=None):
        # Run one simulation step with real dt
        if dt is None:
            dt = self.dt
        self.driver_decision()
        self.move_forward(dt)
        self.step_count += 1
        # Calculate and store inter-vehicular distances for final analysis
        gaps = []
        road_length = self.roads[0].length if self.roads else 1000
        for idx, car in enumerate(self.cars):
            if idx ==0:
                continue
            cars_same_road = [c for c in self.cars if c.current_road == car.current_road and c != car]

            def gap_to(other):
                gap = (car.pos - other.pos ) % road_length
                return gap if gap > 0 else float('inf')
            
            front_car = min(cars_same_road, key=gap_to, default=None) if cars_same_road else None
            gap = ((car.pos - front_car.pos - car.length) % road_length)
            if idx == 0: 
                print(gap)
            if gap > 0:
                gaps.append(gap)
    
        if gaps:
            current_min_gap = min(gaps)
            current_max_gap = max(gaps)
            if current_min_gap < self.overall_min_gap:
                self.overall_min_gap = current_min_gap
            if current_max_gap > self.overall_max_gap:
                self.overall_max_gap = current_max_gap

            
            self.all_gaps.extend(gaps)

    def set_leader_stop(self, leader_stop):
        self.leader_stop = leader_stop

    def set_follower_stop(self, follower_stop):
        self.follower_stop = follower_stop

    # Main code to calculate acclerattion
    def driver_decision(self):
        road_length = self.roads[0].length if self.roads else 1000
        dt = self.dt
        car_states = [(c.pos, c.velocity) for c in self.cars]

        for idx, car in enumerate(self.cars):
            # self.v_des = 0 if (getattr(self, 'follower_stop', False) and idx == 2) else self.initial_v_des

            if idx == 0:
                if getattr(self, 'leader_stop', False):
                    acc = self.kc * (0 - car.velocity)
                    acc = max(self.min_a, min(self.max_a, acc))
                    last_acc = car.acceleration
                    jerk  = (acc - last_acc) / dt
                    max_jerk = 5
                    if jerk > max_jerk:
                        acc = last_acc + max_jerk * dt
                    elif jerk < -max_jerk:
                        acc = last_acc - max_jerk * dt
                    car.acceleration = acc
                    continue

                if hasattr(self, 'lead_velocity_profile') and self.lead_velocity_profile:
                    time = round(self.step_count * dt, 3)
                    for i in range(len(self.lead_velocity_profile)-1):
                        t1,v1 = self.lead_velocity_profile[i]
                        t2, v2 = self.lead_velocity_profile[i+1]
                        if t1 <= time <= t2:
                            alpha = (time - t1) / (t2 - t1)
                            target_velocity = v1 + alpha * (v2 - v1)
                            acc = self.kc * (target_velocity - car.velocity)
                            acc = (target_velocity - car.velocity) / dt
                            acc = max(self.min_a, min(self.max_a, acc))
                            last_acc = car.acceleration
                            jerk  = (acc - last_acc) / dt
                            max_jerk = 5
                            if jerk > max_jerk:
                                acc = last_acc + max_jerk * dt
                            elif jerk < -max_jerk:
                                acc = last_acc - max_jerk * dt
                            car.acceleration = acc
                            break
                    else:
                        # If time exceeds profile, maintain last velocity
                        t1, v1 = self.lead_velocity_profile[-1]
                        target_velocity = v1
                        acc = self.kc * (target_velocity - car.velocity)
                        # acc = (target_velocity - car.velocity) / dt
                        acc = max(self.min_a, min(self.max_a, acc))
                        last_acc = car.acceleration
                        jerk  = (acc - last_acc) / dt
                        max_jerk = 5
                        if jerk > max_jerk:
                            acc = last_acc + max_jerk * dt
                        elif jerk < -max_jerk:
                            acc = last_acc - max_jerk * dt
                        car.acceleration = acc
                else:
                    # Try to reach v_des if no velocity profile
                    acc = self.kc * (self.v_des - car.velocity)
                    acc = max(self.min_a, min(self.max_a, acc))
                    last_acc = car.acceleration
                    jerk  = (acc - last_acc) / dt
                    max_jerk = 5
                    if jerk > max_jerk:
                        acc = last_acc + max_jerk * dt
                    elif jerk < -max_jerk:
                        acc = last_acc - max_jerk * dt
                    car.acceleration = acc
                    continue
                continue
            # if idx == 2:
            #     if getattr(self, 'follower_stop', False):
            #         acc = self.kc * (0 - car.velocity)
            #         acc = max(self.min_a, min(self.max_a, acc))
            #         last_acc = car.acceleration
            #         jerk  = (acc - last_acc) / dt
            #         max_jerk = 5
            #         if jerk > max_jerk:
            #             acc = last_acc + max_jerk * dt
            #         elif jerk < -max_jerk:
            #             acc = last_acc - max_jerk * dt
            #         # car.acceleration = acc
            #         car.acceleration = acc
            #         car.mode = "VEL"
            #         continue
                
            #     if hasattr(self, 'follower_velocity_profile') and self.follower_velocity_profile:
            #         time = round(self.step_count * dt, 3)
            #         for i in range(len(self.follower_velocity_profile)-1):
            #             t1,v1 = self.follower_velocity_profile[i]
            #             t2, v2 = self.follower_velocity_profile[i+1]
            #             if t1 <= time <= t2:
            #                 alpha = (time - t1) / (t2 - t1)
            #                 target_velocity = v1 + alpha * (v2 - v1)
            #                 # current_velocity = car.velocity
            #                 acc = self.kc * (target_velocity - car.velocity)
            #                 # acc = (target_velocity - current_velocity) / dt
            #                 acc = max(self.min_a, min(self.max_a, acc))
            #                 last_acc = car.acceleration
            #                 jerk  = (acc - last_acc) / dt
            #                 max_jerk = 5
            #                 if jerk > max_jerk:
            #                     acc = last_acc + max_jerk * dt
            #                 elif jerk < -max_jerk:
            #                     acc = last_acc - max_jerk * dt
            #                 car.acceleration = acc
            #                 break
            #         else:
            #             # If time exceeds profile, maintain last velocity
            #             t1, v1 = self.follower_velocity_profile[-1]
            #             target_velocity = v1
            #             acc = self.kc * (target_velocity - car.velocity)
            #             # acc = (target_velocity - current_velocity) / dt
            #             acc = max(self.min_a, min(self.max_a, acc))
            #             last_acc = car.acceleration
            #             jerk  = (acc - last_acc) / dt
            #             max_jerk = 5
            #             if jerk > max_jerk:
            #                 acc = last_acc + max_jerk * dt
            #             elif jerk < -max_jerk:
            #                 acc = last_acc - max_jerk * dt
            #             car.acceleration = acc
            #         continue

    
            # Find the car ahead and behind on the same road (circular road)
            
            cars_same_road = [c for c in self.cars if c.current_road == car.current_road and c != car]

            def gap_to(other):
                gap = (car.pos - other.pos ) % road_length
                return gap if gap > 0 else float('inf')
            
            front_car = min(cars_same_road, key=gap_to, default=None) if cars_same_road else None

            def gap_from(other):
                gap = (other.pos - car.pos) % road_length
                return gap if gap > 0 else float('inf')
            
            back_car = min(cars_same_road, key=gap_from, default=None) if cars_same_road else None

            # RL-based Dynamic Weight Control
            if car.weights is not None:
                car.mode = 'RL-W'
                road_length = car.current_road.length
                
                # Retrieve neighbors (handling wrap-around distance)
                # Front Car
                if front_car:
                    front_gap = (front_car.pos - car.pos - car.length) % road_length
                    front_vel = front_car.velocity
                else:
                    front_gap = 1000.0 # Large gap if no car
                    front_vel = car.velocity # No relative velocity
                
                # Back Car    
                if back_car:
                    back_gap = (car.pos - back_car.pos - back_car.length) % road_length
                    back_vel = back_car.velocity
                else:
                    back_gap = 1000.0 
                    back_vel = car.velocity

                # Current State
                ego_vel = car.velocity
                desired_gap = self.min_dis + ego_vel * self.reaction_time
                target_vel = car.target_speed if car.target_speed is not None else self.v_des

                # Weights from RL Agent
                w1, w2, w3, w4, w5 = car.weights
                
                # 5-Term Control Logic
                # 1. Front Gap Error (Maintain distance to front)
                term1 = w1 * (front_gap - desired_gap)
                
                # 2. Back Gap Error (Maintain distance to back)
                term2 = w2 * (desired_gap - back_gap)
                
                # 3. Front Velocity Match
                term3 = w3 * (front_vel - ego_vel)
                
                # 4. Back Velocity Match
                term4 = w4 * (back_vel - ego_vel)
                
                # 5. Target Velocity Match
                term5 = w5 * (target_vel - ego_vel)

                acc = term1 + term2 + term3 + term4 + term5
                acc = max(self.min_a, min(self.max_a, acc))
            
            elif self.model == 'ACC' or ((self.model == 'BCC' or self.model=="ACC+BCC") and idx == len(self.cars) - 1):
                car.mode = 'ACC'
                car_pos, car_vel = car_states[idx]
                front_idx = self.cars.index(front_car)
                front_car_pos, front_car_vel = car_states[front_idx]
                gap = (car_pos - front_car_pos - car.length ) % road_length
                
                rel_v = front_car_vel - car_vel
                desired_gap = self.min_dis + car_vel * self.reaction_time
                acc = self.kd * (gap - desired_gap) +  self.kv * rel_v
                acc = max(self.min_a, min(self.max_a, acc))
            elif self.model == 'BCC':        
                car.mode = 'BCC'        
                car_pos, car_vel = car_states[idx]
                front_idx = self.cars.index(front_car)
                front_car_pos, front_car_vel = car_states[front_idx]
                back_idx = self.cars.index(back_car)
                back_car_pos, back_car_vel = car_states[back_idx]
                desired_gap = self.min_dis + car_vel * self.reaction_time
                front_gap = abs((car_pos - front_car_pos - front_car.length) % road_length)
                back_gap = abs((back_car_pos - car_pos - car.length) % road_length)
                gap_factor = self.kd * (front_gap - desired_gap) + self.kd * (desired_gap - back_gap)
                velocity_factor =  self.kv * (front_car_vel - car_vel) + self.kv * (back_car_vel - car_vel)
                d_vel_factor = 0
                acc = velocity_factor + gap_factor + d_vel_factor
                acc = max(self.min_a, min(self.max_a, acc))

            elif self.model == 'ACC+BCC':
                self.mode = "INTEGRATED"
                ve = car.velocity
                vl = front_car.velocity
                vf = back_car.velocity
                ae = abs(self.min_a)
                af = ae * 0.7  
                Tr = self.reaction_time
                Le = car.length
                Lb = self.min_gap

                Gfront_min = ve * Tr + ((vl - ve) ** 2) / (2 * ae) + Lb
                Grear_min = vf * Tr + ((vf - ve) ** 2) / (2 * af) + Lb

                X = Gfront_min + Le + Grear_min

                car_pos, car_vel = car_states[idx]
                front_idx = self.cars.index(front_car)
                front_car_pos, front_car_vel = car_states[front_idx]
                back_idx = self.cars.index(back_car)
                back_car_pos, back_car_vel = car_states[back_idx]

                front_gap = abs((car_pos - front_car_pos - front_car.length) % road_length)
                back_gap = abs((back_car_pos - car_pos - car.length) % road_length)

                car.integration_factor = self.calculate_integration_factor(front_gap, back_gap, X, car, front_car, back_car, idx)
                iF = car.integration_factor
                desired_gap = self.min_dis + car_vel * self.reaction_time

                front_gap = abs((car_pos - front_car_pos - front_car.length) % road_length)
                back_gap = abs((back_car_pos - car_pos - car.length) % road_length)

                gap_factor = self.kd * (front_gap - desired_gap) + iF  * self.kd * (desired_gap - back_gap)
                velocity_factor =  self.kv * (front_car_vel - car_vel) + iF *  self.kv * (back_car_vel - car_vel)
                d_vel_factor = 0
                # d_vel_factor = self.kc *(self.v_des - car_vel)
                acc = velocity_factor + gap_factor + d_vel_factor
                acc = max(self.min_a, min(self.max_a, acc))

            # Add some hysterises to accleration
            last_acc = car.acceleration
            jerk  = (acc - last_acc) / dt
            max_jerk = 5
            if jerk > max_jerk:
                acc = last_acc + max_jerk * dt
            elif jerk < -max_jerk:
                acc = last_acc - max_jerk * dt
            car.acceleration = acc
            

    def calculate_integration_factor(self, front_gap, back_gap, X, car, front_car, rear_car, idx):
        """
        Advanced integration factor:
          - More factors: gaps, relative speeds, rear car behavior
          - Smooth hysteresis
        """
        

        # if back_gap > X :
        #     car.mode = 'ACC'
        #     return 0
        # else:
        #     car.mode = 'BCC'
        #     return 1
        


        total_w = 0
        raw_iF = 0
        back_ratio_w = 6
        front_ratio_w = 6
        front_ratio_validation_threshold = X
        rear_brake_ratio_w = 2
        closing_ratio_back_w = 2
        closing_ratio_front_w = 3
        rel_vel_validation_threshold = 2 * X  

        # 1) Gap ratios
        # Normalise all the factors to [0 , 1]

        # 1.1) Back Ratio
        back_ratio = (X - min(X, back_gap)) / X
        total_w +=back_ratio_w
        raw_iF += back_ratio * back_ratio_w


        # 1.2) Front Ratio
        # if back_gap < front_ratio_validation_threshold:
        g_threshold = max(10, car.velocity * car.headway_time)
        g_threshold = 10
        front_ratio = math.exp(-0.01* math.pow((front_gap - g_threshold), 2))
        raw_iF += front_ratio * front_ratio_w
        total_w += front_ratio_w
        # else:
            # front_ratio = 0


       
        # 2) Rear braking
        # If acc>2(Comfort limit) then shift to acc else prefer bcc
        # The function returns 1 for greater than -2 or else it exponentially drops to 0
        def smooth_rear_brake_factor(k,a):
            if a > -2:
                return 1.0
            else:
                return math.exp(k * (a + 2))

        if back_gap < X and rear_car.acceleration <0:
            rear_brake_ratio = smooth_rear_brake_factor(1.23, rear_car.acceleration)
            total_w +=rear_brake_ratio_w
            raw_iF += rear_brake_ratio * rear_brake_ratio_w
        else:
            rear_brake_ratio = 0.0


        # 3) Relative rear velocity
        rel_vel_rear = rear_car.velocity - car.velocity
        if back_gap < rel_vel_validation_threshold and rel_vel_rear > 0:
            closing_in_ratio = np.clip(rel_vel_rear / 5.0, 0, 1)
            total_w += closing_ratio_back_w
            raw_iF += closing_in_ratio * closing_ratio_back_w
        else:
            closing_in_ratio = 0.0

        
        # 4) Relative front velocity
        rel_vel_front = car.velocity - front_car.velocity
        if front_gap < rel_vel_validation_threshold and rel_vel_front > 0:
            closing_ratio_front = np.clip(rel_vel_front / 5.0, 0, 1)
            raw_iF += (closing_ratio_front * closing_ratio_front_w)
            total_w +=closing_ratio_front_w
        else:
            closing_ratio_front = 0.0


        # 5) Weighted sum
        normalized_iF = raw_iF / total_w

        # 6) Clamp to [0, 1]
        normalized_iF = min(1, max(0, normalized_iF))

        # 7) Smooth with hysteresis
        old_iF = getattr(car, 'integration_factor', 0)
        alpha = 0.009
        smoothed_iF = (1 - alpha) * old_iF + alpha * normalized_iF
        car.integration_factor = smoothed_iF
        # if idx == 1:
        #     print(f"Time: {(self.dt * self.step_count):.2f},b_r: {back_ratio:.3f}, f_r: {front_ratio:.3f}, r_b_r: {rear_brake_ratio:.3f}, c_f_r: {closing_ratio_front:.3f}, c_i_r: {closing_in_ratio:.3f},T_Weight: {total_w} ,IF: {car.integration_factor:.2f}")

        # 8) Mode switch for visualization
        if smoothed_iF < 0.1:
            # smoothed_iF = 0
            car.mode = 'ACC'
        elif smoothed_iF > 0.8:
            car.mode = 'BCC'
        else:
            car.mode = 'INTEGRATED'

        # 9) Return the value
        return smoothed_iF

    def move_forward(self, dt=None):
        if dt is None:
            dt = self.dt
        # Move all cars forward based on their velocity and acceleration
        for car in self.cars:
            car.update(dt)
            # Clamp velocity to not exceed max_v
            car.velocity = max(self.min_v, min(car.velocity, self.max_v))

            # Fade collision color if timer is active
            if hasattr(car, 'collision_timer') and car.collision_timer > 0:
                car.collision_timer -= 1
                if car.collision_timer == 0:
                    car.color = car.original_color
        # Handle any collisions that may have occurred
        self.handle_collisions()

    def handle_collisions(self):
        # Sort cars by position to check for overlaps (circular road)
        road_length = self.roads[0].length if self.roads else 1000
        sorted_cars = sorted(self.cars, key=lambda c: c.pos)
        for i, car in enumerate(sorted_cars):
            next_car = sorted_cars[(i + 1) % len(sorted_cars)]
            if car == next_car:
                continue
            min_gap = self.min_gap  # Use instance parameter

            # Calculate gap considering circular road
            hasCollided = ((next_car.pos - car.pos) % road_length) <= car.length
            if hasCollided:
                v1 = car.velocity
                v2 = next_car.velocity
                e = car.CoR if hasattr(car, 'CoR') else 0.3

                # Prevent overlap: set next_car just behind car with an addition of min_gap
                car.velocity = ((1 - e) * v1 + (1 + e) * v2) / 2
                next_car.velocity = ((1 - e) * v2 + (1 + e) * v1) / 2

                next_car.pos = (car.pos + car.length + min_gap) % road_length

                # Change color to indicate collision and start timer
                car.color = 'orange'
                next_car.color = 'orange'
                car.collision_timer = 40  # 40 steps * 0.1s = 4 seconds
                next_car.collision_timer = 40
    
     