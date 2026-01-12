import jax.numpy as jnp
import jax
import optax
from model import get_ppvs
from tqdm import tqdm
import numpy as np

def optimize(init_params, ppvs_target, model_args, learning_rate, params_history=None, loss_target=1e-6):

    def get_output(params):

        output, dvs, ems, models = get_ppvs(params, model_args)

        return output, dvs, ems, models

    def loss_fn(params, target):

        output, dvs, _, _ = get_output(params)

        lambda0 = 1e0
        lambda1 = 1e0

        mask = jnp.ones_like(output)

        lambda_penalty = 0e0
        penalty = jnp.mean(jnp.square(jnp.minimum(-dvs, 0.0)))

        total_loss = lambda0 * optax.l2_loss(output[0] * mask[0], target[0] * mask[0]) + \
                        lambda1 * optax.l2_loss(output[1] * mask[1], target[1] * mask[1])

        return jnp.mean(total_loss) + penalty * lambda_penalty

    loss_and_grad = jax.value_and_grad(loss_fn)
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(init_params)

    @jax.jit
    def step(params, opt_state, target):
        loss, grads = loss_and_grad(params, target)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss


    params = init_params
    loss_history = []
    pbar = tqdm(total=100)

    params, opt_state, loss = step(params, opt_state, ppvs_target)

    # this is for progress bar tqdm
    loss_init = loss
    loss_old = loss

    icount = 0
    while True:
        params, opt_state, loss = step(params, opt_state, ppvs_target)
        if icount % 10 == 0:
            if params_history is not None:
                for k in params.keys():
                    params_history[k].append(params[k])
            #print(f"Step {i}/{epochs}, Loss: {loss:.6e}")
            pbar.set_description(f"Loss: {loss:.6e}, Target: {loss_target:.6e}, Epoch: {icount}")
            dd = (np.log10(1/loss_target) - np.log10(1/loss_old)) / (np.log10(1/loss_target) - np.log10(1/loss_init)) * pbar.total
            pbar.n = np.round(min(pbar.total - dd, pbar.total), decimals=3)
            pbar.refresh()
            loss_old = loss
            #print(f"Step {i}/{epochs}, Loss: {loss:.6e} (threshold: {loss_target:.6e})")
        loss_history.append(loss)
        if loss < loss_target:
            print(f"Converged at step {icount}, Loss: {loss:.6e} < {loss_target:.6e}")
            break
        icount += 1
    pbar.n = pbar.total
    pbar.close()

    output, _, ems, models = get_output(params)

    return params, output, ems, models, params_history, loss_history
