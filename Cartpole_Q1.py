import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import time
import numpy as np

# Crear el entorno original
env = gym.make("CartPole-v1")

# Modificar el entorno para penalizar al agente por acercarse a los límites
class CustomCartPoleEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = gym.make("CartPole-v1")
        
        # Definir los espacios de observación y acción
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        # Obtener la posición 'x' del carro (primer valor en la observación)
        x = obs[0]

        # Penalizar si el carro se acerca mucho a los límites
        if abs(x) > 2.0:  # Si está muy cerca de los límites
            reward -= 1  # Penalización adicional por acercarse al borde

        # Si el carro se sale del límite, el episodio termina como antes
        if abs(x) >= 2.4:
            done = True

        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        return self.env.reset()

    def render(self, **kwargs):
        return self.env.render()

    def close(self):
        self.env.close()

# Crear el entorno con las nuevas reglas de penalización
env = CustomCartPoleEnv()

# Crear el modelo DQN
model = DQN(
    "MlpPolicy",  # Política basada en redes neuronales densas
    env,
    learning_rate=1e-3,
    buffer_size=10000,
    learning_starts=1000,
    batch_size=64,
    gamma=0.99,
    target_update_interval=500,
    train_freq=4,
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
    verbose=1,
)

# Entrenamiento del modelo
print("Entrenando el modelo...")
model.learn(total_timesteps=50000)

# Evaluar el modelo
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Recompensa media después del entrenamiento: {mean_reward} ± {std_reward}")

# Guardar el modelo
model.save("dqn_cartpole")
print("Modelo guardado como 'dqn_cartpole.zip'.")

# Visualización del agente entrenado
print("Probando el modelo entrenado...")

env = gym.make("CartPole-v1", render_mode="human")
obs, _ = env.reset()
done = False


while not done:
    action, _ = model.predict(obs, deterministic=True)  # Predecir la acción
    obs, reward, done, truncated, _ = env.step(action)  # Ejecutar la acción
    time.sleep(0.05)  # Pausar entre pasos para ver la simulación
    if truncated:  # Si el episodio termina por truncamiento, sale del bucle
        break

env.close()  # Cierra el entorno después de terminar
