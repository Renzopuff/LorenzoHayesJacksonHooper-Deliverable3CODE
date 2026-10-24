from opacus.accountants import RDPAccountant, PRVAccountant

q, T = 0.02, 300
for target, sigma in [(1, 1.6797), (2, 1.1206), (4, 0.8350)]:
    rdp, prv = RDPAccountant(), PRVAccountant()
    for _ in range(T):
        rdp.step(noise_multiplier=sigma, sample_rate=q)
        prv.step(noise_multiplier=sigma, sample_rate=q)
    for d in [1e-2, 1e-3, 1e-5]:
        print(f"target={target} sigma={sigma} delta={d:g} "
              f"RDP={rdp.get_epsilon(d):.3f} PRV={prv.get_epsilon(d):.3f}")