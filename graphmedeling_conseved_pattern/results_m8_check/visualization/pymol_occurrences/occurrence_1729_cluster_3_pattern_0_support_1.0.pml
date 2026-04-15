load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7wnm.ent", occ_1729_c3_p0_s1.0
hide everything, occ_1729_c3_p0_s1.0
show cartoon, occ_1729_c3_p0_s1.0 and chain B+A
color palegreen, occ_1729_c3_p0_s1.0 and chain B
color lightblue, occ_1729_c3_p0_s1.0 and chain A
select hotspot_source, occ_1729_c3_p0_s1.0 and ((chain B and resi 83))
select hotspot_target, occ_1729_c3_p0_s1.0 and ((chain A and resi 487))
select hotspot_all, occ_1729_c3_p0_s1.0 and ((chain A and resi 487) or (chain B and resi 83))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_1729_c3_p0_s1.0 and chain B+A
set_name hotspot_all, hotspot_occurrence_1729
set_name hotspot_source, hotspot_source_1729
set_name hotspot_target, hotspot_target_1729
bg_color white
# patternId=0 support=1.0 graphId=282
