# Action diffusion lab: two paths, one obstacle

The exercise isolates one reason to use a generative action model. The demonstrations contain two safe trajectories around an obstacle. Squared-error regression predicts their mean, which collides. A conditional diffusion model should recover both modes.

## What participants implement

There are three bounded TODOs:

1. the closed-form forward noising process, `q_sample`;
2. the noise-prediction training loss, `diffusion_loss`;
3. one DDPM reverse step, `p_sample`.

The notebook supplies the dataset, conditional denoiser, training loop, sampler, plots, metrics, and checks. The point is to understand the mechanics of an action diffusion model, not to spend the session on scaffolding.

## What success looks like

The final evaluation compares the learned sampler with the conditional-mean baseline. A successful run should show:

- baseline collision rate near 100%;
- diffusion collision rate below 20%;
- endpoint error near zero because the boundary is enforced;
- both upper and lower modes represented.

## Running it

Open `diffusion_policy_lab_student.ipynb` in Google Colab or Jupyter. A standard Colab runtime already contains the required packages. Locally:

```bash
python -m pip install -r requirements.txt
jupyter lab diffusion_policy_lab_student.ipynb
```

Expected working time is 70–85 minutes.

## Turn-in

Submit one figure containing:

- expert demonstrations;
- the conditional-mean regression baseline;
- diffusion samples.

Add three sentences: why regression collides, how diffusion retains both modes, and which metric would have exposed a failed solution.

