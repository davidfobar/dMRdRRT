# dMRdRRT

Toy implementation of Rapidly-Exploring Random Trees (RRT) for 2D path planning.

## Run the demo

```bash
python main.py
```

This runs a deterministic toy planning problem with circular obstacles, prints path stats, and saves:

- `images/rrt_toy_solution.png`

To open the plot window while also saving the image:

```bash
python main.py --show
```

To enable RRT* (cost-based parent selection + local rewiring):

```bash
python main.py --rrt-star
```