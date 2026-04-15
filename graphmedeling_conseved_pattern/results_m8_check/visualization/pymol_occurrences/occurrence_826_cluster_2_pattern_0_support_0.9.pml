load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7dhx.ent", occ_826_c2_p0_s0.9
hide everything, occ_826_c2_p0_s0.9
show cartoon, occ_826_c2_p0_s0.9 and chain A+B
color palegreen, occ_826_c2_p0_s0.9 and chain A
color lightblue, occ_826_c2_p0_s0.9 and chain B
select hotspot_source, occ_826_c2_p0_s0.9 and ((chain A and resi 30))
select hotspot_target, occ_826_c2_p0_s0.9 and ((chain B and resi 417))
select hotspot_all, occ_826_c2_p0_s0.9 and ((chain A and resi 30) or (chain B and resi 417))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_826_c2_p0_s0.9 and chain A+B
set_name hotspot_all, hotspot_occurrence_826
set_name hotspot_source, hotspot_source_826
set_name hotspot_target, hotspot_target_826
bg_color white
# patternId=0 support=0.9 graphId=34
