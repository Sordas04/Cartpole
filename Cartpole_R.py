import gymnasium as gym
import numpy as np
import time


# Crear el entorno
env = gym.make("CartPole-v1", render_mode="human")

# Parámetros de entrenamiento
num_episodios = 1000
max_pasos = 200  # Número máximo de pasos por episodio

# Política aleatoria
def politica_aleatoria(observacion):
    # Devuelve una acción aleatoria: 0 (izquierda) o 1 (derecha)
    return env.action_space.sample()

# Entrenamiento del agente
for episodio in range(num_episodios):
    observacion, _ = env.reset()  # Reiniciar el entorno
    recompensa_total = 0

    for t in range(max_pasos):
        env.render()  # Renderizar el entorno
        accion = politica_aleatoria(observacion)  # Seleccionar una acción
        observacion, recompensa, terminado, truncado, _ = env.step(accion)
        recompensa_total += recompensa

        if terminado or truncado:
            break

    print(f"Episodio {episodio + 1}: Recompensa total = {recompensa_total}")

env.close()
