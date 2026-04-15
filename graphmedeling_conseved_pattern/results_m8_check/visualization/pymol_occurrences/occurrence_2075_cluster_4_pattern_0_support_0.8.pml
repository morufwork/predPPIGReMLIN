load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7sy6.ent", occ_2075_c4_p0_s0.8
hide everything, occ_2075_c4_p0_s0.8
show cartoon, occ_2075_c4_p0_s0.8 and chain B+E
color palegreen, occ_2075_c4_p0_s0.8 and chain B
color lightblue, occ_2075_c4_p0_s0.8 and chain E
select hotspot_source, occ_2075_c4_p0_s0.8 and ((chain B and resi 486))
select hotspot_target, occ_2075_c4_p0_s0.8 and ((chain E and resi 82) or (chain E and resi 83))
select hotspot_all, occ_2075_c4_p0_s0.8 and ((chain B and resi 486) or (chain E and resi 82) or (chain E and resi 83))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_2075_c4_p0_s0.8 and chain B+E
set_name hotspot_all, hotspot_occurrence_2075
set_name hotspot_source, hotspot_source_2075
set_name hotspot_target, hotspot_target_2075
bg_color white
# patternId=0 support=0.8 graphId=212
