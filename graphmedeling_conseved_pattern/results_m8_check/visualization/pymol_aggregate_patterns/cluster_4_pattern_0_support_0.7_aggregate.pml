load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7dhx.ent", agg_c4_p0_s0.7
hide everything, agg_c4_p0_s0.7
show cartoon, agg_c4_p0_s0.7 and chain A+B
color palegreen, agg_c4_p0_s0.7 and chain A
color lightblue, agg_c4_p0_s0.7 and chain B
select hotspot_source, agg_c4_p0_s0.7 and ((chain A and resi 27) or (chain A and resi 353))
select hotspot_target, agg_c4_p0_s0.7 and ((chain B and resi 456) or (chain B and resi 475) or (chain B and resi 489) or (chain B and resi 505))
select hotspot_all, agg_c4_p0_s0.7 and ((chain A and resi 27) or (chain B and resi 456) or (chain B and resi 475) or (chain B and resi 489) or (chain A and resi 353) or (chain B and resi 505))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient agg_c4_p0_s0.7 and chain A+B
bg_color white
set_name hotspot_all, aggregate_hotspot_4_0
set_name hotspot_source, aggregate_source_4_0
set_name hotspot_target, aggregate_target_4_0
# aggregate top residues for cluster=4 patternId=0 support=0.7
