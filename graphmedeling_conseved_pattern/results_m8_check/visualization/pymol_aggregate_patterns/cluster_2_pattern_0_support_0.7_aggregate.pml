load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb6lzg.ent", agg_c2_p0_s0.7
hide everything, agg_c2_p0_s0.7
show cartoon, agg_c2_p0_s0.7 and chain A+B
color palegreen, agg_c2_p0_s0.7 and chain A
color lightblue, agg_c2_p0_s0.7 and chain B
select hotspot_source, agg_c2_p0_s0.7 and ((chain A and resi 31))
select hotspot_target, agg_c2_p0_s0.7 and ((chain B and resi 484))
select hotspot_all, agg_c2_p0_s0.7 and ((chain A and resi 31) or (chain B and resi 484))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient agg_c2_p0_s0.7 and chain A+B
bg_color white
set_name hotspot_all, aggregate_hotspot_2_0
set_name hotspot_source, aggregate_source_2_0
set_name hotspot_target, aggregate_target_2_0
# aggregate top residues for cluster=2 patternId=0 support=0.7
