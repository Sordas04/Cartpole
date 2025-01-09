import gymnasium as gym
import time

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

# Crear el entorno
env = gym.make("CartPole-v1")

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
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, _ = env.step(action)
    time.sleep(0.05)
    if truncated:
        break

env.close()

