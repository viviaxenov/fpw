from fpw.utility import *

import matplotlib.pyplot as plt
import corner


# thetas = torch.Tensor([i*2.*torch.pi/5. for i in range(5)])
# ms = torch.stack((torch.cos(thetas), torch.sin(thetas)), axis=-1)
# 
# mix = torch.distributions.Categorical(torch.ones(5,))
# comp = torch.distributions.Independent(torch.distributions.Normal(
#          ms, torch.full((5,2), fill_value=0.2)), 1)
# gmm = torch.distributions.MixtureSameFamily(mix, comp)
# 
# d = gmm
d = Nonconvex(a=np.array([ -1., 0.,]))

mala_step = MALAStep(d, 0.05)
x0 = torch.randn(200, 2)
mala_step.tune(x0)

xr = d.sample((200,), run_mcmc=True)
xr = d.sample(200, run_mcmc=True)
x1 = x0.detach().clone()
for _ in range(200):
    x1 = mala_step(x1)

x0 = x0.numpy()
x1 = x1.numpy()
xr = xr.numpy()

plt.scatter(*x0.T, label="Initial")
plt.scatter(*x1.T, label="Approx.")
plt.scatter(*xr.T, label="Ref.")

plt.legend()
plt.show()


exit()

s = d.sample(N_samples=1000, run_mcmc=True)
s = d.sample(N_samples=2000, run_mcmc=True)
corner.corner(s, bins=20, smooth=0.5, smooth1d=0.3)
plt.show()
