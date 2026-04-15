load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7efr.ent", occ_149_c0_p0_s0.9
hide everything, occ_149_c0_p0_s0.9
show cartoon, occ_149_c0_p0_s0.9 and chain A+B
color palegreen, occ_149_c0_p0_s0.9 and chain A
color lightblue, occ_149_c0_p0_s0.9 and chain B
select hotspot_source, occ_149_c0_p0_s0.9 and ((chain A and resi 330))
select hotspot_target, occ_149_c0_p0_s0.9 and ((chain B and resi 500))
select hotspot_all, occ_149_c0_p0_s0.9 and ((chain A and resi 330) or (chain B and resi 500))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_149_c0_p0_s0.9 and chain A+B
set_name hotspot_all, hotspot_occurrence_149
set_name hotspot_source, hotspot_source_149
set_name hotspot_target, hotspot_target_149
bg_color white
# patternId=0 support=0.9 graphId=82
