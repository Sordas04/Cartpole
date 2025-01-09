import gymnasium as gym
import numpy as np
import time
# Definir el regulador PID

    
def pid_control(theta, theta_dot, integral_error, kp, ki, kd, dt):
    """
    Regulador PID para el ángulo del péndulo.

    Parameters:
        theta (float): Ángulo actual del péndulo (en radianes).
        theta_dot (float): Velocidad angular del péndulo (en rad/s).
        integral_error (float): Error acumulado para el término integral.
        kp (float): Ganancia proporcional.
        ki (float): Ganancia integral.
        kd (float): Ganancia derivativa.
        dt (float): Intervalo de tiempo entre pasos.

    Returns:
        force (float): Fuerza calculada para estabilizar el sistema.
    """
    # Cálculo del control PID
    derivative = theta_dot
    force = (kp * theta)
    return force

# Inicializar el entorno
env = gym.make("CartPole-v1", render_mode="human")
env.reset()

# Configuración del PID
kp = 30.0   # Ganancia proporcional
ki = 5.52     # Ganancia integral
kd = 3.66    # Ganancia derivativa
dt = 0.02    # Intervalo de tiempo

# Variables iniciales
integral_error = 0.0


episode_count = 4
for episode in range(episode_count):
    # Reiniciar el entorno y obtener el estado inicial
    state = env.reset()[0]
    
    integral_error = 0.0   # Reiniciar error acumulado
    done = False


    while not done:
        # Extraer variables relevantes del estado
        cart_position, cart_velocity, pole_angle, pole_angular_velocity = state
        
        # Acumular el error para el término integral
        integral_error += pole_angle

        # Calcular la fuerza usando el regulador PID
        force = pid_control(pole_angle, pole_angular_velocity, integral_error, kp, ki, kd, dt)

        # Discretizar la fuerza para que sea compatible con el entorno (acción: -1 o 1)
        action = 1 if force > 0 else 0

        # Aplicar la acción y obtener el siguiente estado
        state, reward, done, truncated, info = env.step(action)
        
        time.sleep(0.05)
        
        # Terminar si el episodio se completa
        if done or truncated:
            break

env.close()





