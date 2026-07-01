# RL MLOps Pipeline
 
Pipeline completo de MLOps para entrenar, validar y desplegar agentes de Reinforcement Learning en un entorno GridWorld 20×20.
 
## Algoritmos
 
Q-Learning tabular · DQN (PyTorch) · Discrete SAC (PyTorch)
 
## Escenarios
 
`wall` · `l_shape` · `u_shape`
 
## Stack
 
| Pieza | Tecnología |
|---|---|
| Datos | DVC + 41 experimentos reales |
| Quality gates | pytest (83 tests) |
| CI/CD | GitHub Actions + CML (self-hosted runner) |
| Tracking | MLflow |
| API | FastAPI + Server-Sent Events |
| Frontend | React + Vite |
