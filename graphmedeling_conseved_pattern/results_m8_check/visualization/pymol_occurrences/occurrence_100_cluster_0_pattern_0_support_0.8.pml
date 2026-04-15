load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7sy0.ent", occ_100_c0_p0_s0.8
hide everything, occ_100_c0_p0_s0.8
show cartoon, occ_100_c0_p0_s0.8 and chain B+E
color palegreen, occ_100_c0_p0_s0.8 and chain B
color lightblue, occ_100_c0_p0_s0.8 and chain E
select hotspot_source, occ_100_c0_p0_s0.8 and ((chain B and resi 498))
select hotspot_target, occ_100_c0_p0_s0.8 and ((chain E and resi 41))
select hotspot_all, occ_100_c0_p0_s0.8 and ((chain B and resi 498) or (chain E and resi 41))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_100_c0_p0_s0.8 and chain B+E
set_name hotspot_all, hotspot_occurrence_100
set_name hotspot_source, hotspot_source_100
set_name hotspot_target, hotspot_target_100
bg_color white
# patternId=0 support=0.8 graphId=206
