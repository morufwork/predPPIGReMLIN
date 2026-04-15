load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb6lzg.ent", agg_c1_p0_s0.9
hide everything, agg_c1_p0_s0.9
show cartoon, agg_c1_p0_s0.9 and chain A+B
color palegreen, agg_c1_p0_s0.9 and chain A
color lightblue, agg_c1_p0_s0.9 and chain B
select hotspot_source, agg_c1_p0_s0.9 and ((chain A and resi 41))
select hotspot_target, agg_c1_p0_s0.9 and ((chain B and resi 500))
select hotspot_all, agg_c1_p0_s0.9 and ((chain A and resi 41) or (chain B and resi 500))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient agg_c1_p0_s0.9 and chain A+B
bg_color white
set_name hotspot_all, aggregate_hotspot_1_0
set_name hotspot_source, aggregate_source_1_0
set_name hotspot_target, aggregate_target_1_0
# aggregate top residues for cluster=1 patternId=0 support=0.9
