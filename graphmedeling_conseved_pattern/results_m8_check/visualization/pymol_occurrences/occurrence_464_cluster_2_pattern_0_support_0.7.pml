load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7sy0.ent", occ_464_c2_p0_s0.7
hide everything, occ_464_c2_p0_s0.7
show cartoon, occ_464_c2_p0_s0.7 and chain B+E
color palegreen, occ_464_c2_p0_s0.7 and chain B
color lightblue, occ_464_c2_p0_s0.7 and chain E
select hotspot_source, occ_464_c2_p0_s0.7 and ((chain B and resi 417))
select hotspot_target, occ_464_c2_p0_s0.7 and ((chain E and resi 30))
select hotspot_all, occ_464_c2_p0_s0.7 and ((chain B and resi 417) or (chain E and resi 30))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_464_c2_p0_s0.7 and chain B+E
set_name hotspot_all, hotspot_occurrence_464
set_name hotspot_source, hotspot_source_464
set_name hotspot_target, hotspot_target_464
bg_color white
# patternId=0 support=0.7 graphId=200
