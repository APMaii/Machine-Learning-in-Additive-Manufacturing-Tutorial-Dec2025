'''

Some fucntions for importing in another script .py

'''

def newton(mass , acceleration):
    
    force = mass * acceleration
    
    return force



def check_temperature(temp):
    
    if temp > 2000:
        return True
    else:
        return False
    
    
def check_speed_and_tmperature(temp , speed):
    
    if temp>2000 and speed>400:
        print('warning')
    else:
        print('ok suitable')


'''
e=p/vht
power
speed
hatch
thickness

power , speed, hatch , thcikness --> [] --> e


'''




def VED_calculator(power, speed , hatch , thcikness):
    
    #if thciknes >1000:
        #thcikness/1000
        
    #if speed > 
    #return None error ...
    
    base = speed * hatch * thcikness
    
    ved = power / base
    
    return ved





def sqrt(adad):
    radical = adad
    return radical







    
