import secrets

import mlx.core as mx
import mlx_lm.sample_utils as su
import mlx_lm.server as srv


# mlx_lm.server samples from a thread-local RNG state that restarts identically on every request (and ignores `seed`),
# so repeated requests return the same sample. Draw from a fresh explicit key instead; temperature 0 is unaffected.
def categorical_sampling(logits, temp):
    return mx.random.categorical(logits * (1 / temp), key=mx.random.key(secrets.randbits(31)))


su.categorical_sampling = categorical_sampling
srv.main()
