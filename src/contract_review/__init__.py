# torch doit être importé avant faiss : dans l'ordre inverse, les deux runtimes OpenMP
# entrent en conflit et les calculs de torch se bloquent (observé sur macOS).
import torch  # noqa: F401
