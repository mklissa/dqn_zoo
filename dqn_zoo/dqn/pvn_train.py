
import chex
import jax
import jax.numpy as jnp
import optax


def pvn_train_step(
    pvn_state,
    indicator_state,
    batch,
):

  chex.assert_rank([
      batch.s_tm1, batch.s_t
  ], [4, 4])

  def loss_fn(
      dsm_params, indicator_params
  ):
    indicators = indicator_state.apply_fn(indicator_params, batch.s_tm1)
    rewards = indicators.rewards
    rewards = jax.lax.stop_gradient(rewards)

    # Perform forward passes
    cur_values = pvn_state.apply_fn(dsm_params, batch.s_tm1).predictions
    target_values = pvn_state.apply_fn(pvn_state.target_params,
                                         batch.s_t).predictions

    # === DSM Loss ===
    td_targets = rewards + (0.99**5) * target_values
    td_targets = jax.lax.stop_gradient(td_targets)

    # TD errors
    td_errors = td_targets - cur_values
    chex.assert_rank(td_errors, 2)
    tde_loss = jnp.mean(optax.l2_loss(td_errors))

    loss = tde_loss

    # === Quantile Regression Loss ===
    target_reward_proportion = 0.1
    dsm_pre_rewards = indicators.pre_threshold
    proportion_loss = dsm_pre_rewards * ((1.0 - target_reward_proportion) -
                                         (dsm_pre_rewards < 0.0))
    chex.assert_rank(proportion_loss, 2)
    proportion_loss = jnp.mean(proportion_loss)

    # Mask out the TDE loss if we haven't taken enough warmup steps
    # loss += proportion_loss
    # loss *= (indicator_state.step > 5_000)
    # loss += (indicator_state.step < 5_000) * proportion_loss
    loss += proportion_loss

    return loss

  train_grads, indicator_grads = jax.grad(loss_fn, argnums=(0, 1))(
      pvn_state.params, indicator_state.params)

  pvn_state = pvn_state.apply_gradients(grads=train_grads)
  indicator_state = indicator_state.apply_gradients(grads=indicator_grads)

  return pvn_state, indicator_state

