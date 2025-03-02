# Self-Adaptive Quantum-Classical Evolutionary Optimization

Hey there! This project merges quantum-inspired ideas with classical evolutionary strategies—think rotation gates colliding with mutation/crossover, orchestrated by meta-controllers that adapt themselves. Sound wild? It is! This approach aims to tackle the classic Sphere Function (summing squares) and push fitness below \(10^{-6}\).

> **Curious about the deeper theory, experiments, and math behind this approach?**  
> Check out the detailed paper on ResearchGate:  
> [Self-Adaptive Quantum-Classical Evolutionary Optimization Using Meta-Evolutionary Controllers for the Sphere Function](https://www.researchgate.net/publication/389499115_Self-Adaptive_Quantum-Classical_Evolutionary_Optimization_Using_Meta-Evolutionary_Controllers_for_the_Sphere_Function)

## Why It’s Cool
1. **Quantum + Evolution**: We fuse quantum rotation gates with classical EAs to balance exploration (quantum) and exploitation (classical).
2. **Meta-Evolution**: Our algorithm doesn’t just evolve solutions, it also evolves the *controllers* (parameters) that guide the main population—like AI babysitting AI.
3. **Self-Adaptation**: You don’t just fix or guess parameters; they adapt over time and (hopefully) keep you at the edge of convergence success.

## Project Structure
- **`sphereFunc(x)`**: The Sphere objective function.  
- **`evolveSubset(...)`**: Handles the quantum step (rotation gate) and the classical step (mutation, crossover) for one chunk of the population.  
- **`evolveMetaPopulation(...)`**: Tweaks the meta-controllers themselves to avoid stagnation.  
- **Main Script**: Ties everything together, running the evolutionary loop, logging progress, and generating plots.

## How to Run
1. Clone or download this repository.  
2. Open MATLAB/Octave and run the main script (e.g., `main.m`).
3. Check the console for generation-by-generation feedback, and keep an eye out for re-init events and final best fitness.

## Visualizations
We’ve thrown in plots to track:
- Best vs. average fitness over generations.  
- Re-inits (cool black dots on the best-fitness curve).  
- The average of each meta-parameter (quantum weight, classical weight, adaptation rate) over time.

## Questions / Concerns
Open an issue or ping me with questions. The code’s not perfect—no code is—but it’s a rad playground for optimization fanatics.

## License
TBD or insert your license text here.

---

_If you’re feeling adventurous, fork away and hack on the code. Let’s see if it actually hits \(10^{-6}\) without losing its quantum marbles!_
