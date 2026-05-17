This is the repo for the the group project for the course "Artificial Intelligence 2".
 

## Notes on the Tasks for A (i, ii and iii)
Since the goal is to compare diffusion to standard behavioral cloning, we will need to train a different BC and diffusion policy for each **task**.
The first and most basic could be the double pendulum swing-up task, which has 2 degrees of freedom and therefore can be easily visualized and understood. 

A list of potential tasks to consider:
- Threading (Shows the resistance to error accumulation since the thread can easily get off track)
- Stacking (Shows the ability to recover from mistakes since the stack can easily topple)
- Coffe prep (Shows the ability to handle long-horizon tasks with multiple steps)