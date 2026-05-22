import fasteigenpy as eigen
from unfolding.histogram import UnfoldedHistogram, get_genreco_name
from unfolding.specs import NuisanceTreatment

uh = UnfoldedHistogram.from_disk('/eos/user/d/dponman/proj_unfold_workspace/unfolded')
uh.compute_invhess()
h = uh.to_basic_histogram(NuisanceTreatment(profile=[], fix=[], fixvals=[], num=0))

print('axis_names:', h.binning.axis_names)
print('n_blocks:', len(h.binning._blocks))
for i, b in enumerate(h.binning._blocks):
    print(f'Block {i}: axes={b.axis_names} extents={b.extents} total_size={b.total_size} offset={b.offset}')
    for n in b.axis_names:
        e = b.ax_details[n]['edges']
        print(f'  {n}: [{e[0]:.4g} ... {e[-1]:.4g}] ({len(e)-1} bins)')
