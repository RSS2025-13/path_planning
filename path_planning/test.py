import numpy as np
def pose_callback(points,location):
    #where should this be?
    num_points = len(points)
    distances = np.zeros(num_points-1)#distance from location to line segment, defined by first point
    for i in range(num_points-1):
        a1 = location[:2] - points[i]
        a2 = points[i+1] - points[i]
        L = dist(a2,0)
        if L > 1e-5:
            coeff = max(0,min(1,dot_prod(a1,a2)/L))
            proj_point = points[i]+coeff*a2
        else:
            proj_point = points[i]
        distances[i] = dist(proj_point,location[:2])
    closest_index = np.argmin(distances)
    segment = points[closest_index:closest_index+2]
    return segment

def dot_prod(p1,p2):
        #numpy operations
        return sum(np.multiply(p1,p2))
    
def dist(p1,p2):
    #numpy operations
    return sum(np.square(p1-p2))**0.5

a = np.random.randint(0,100,(100000,2))
# for p in a:
#      print(f'({p[0]}, {p[1]})')
loc = np.array([-1000,23525])
print(pose_callback(a,loc))