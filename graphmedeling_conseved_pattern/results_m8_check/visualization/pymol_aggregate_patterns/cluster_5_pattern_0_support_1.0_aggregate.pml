load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb6lzg.ent", agg_c5_p0_s1.0
hide everything, agg_c5_p0_s1.0
show cartoon, agg_c5_p0_s1.0 and chain A+B
color palegreen, agg_c5_p0_s1.0 and chain A
color lightblue, agg_c5_p0_s1.0 and chain B
select hotspot_source, agg_c5_p0_s1.0 and ((chain A and resi 27))
select hotspot_target, agg_c5_p0_s1.0 and ((chain B and resi 456) or (chain B and resi 489))
select hotspot_all, agg_c5_p0_s1.0 and ((chain A and resi 27) or (chain B and resi 456) or (chain B and resi 489))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient agg_c5_p0_s1.0 and chain A+B
bg_color white
set_name hotspot_all, aggregate_hotspot_5_0
set_name hotspot_source, aggregate_source_5_0
set_name hotspot_target, aggregate_target_5_0
# aggregate top residues for cluster=5 patternId=0 support=1.0
