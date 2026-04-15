load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb8dma.ent", occ_591_c2_p0_s0.7
hide everything, occ_591_c2_p0_s0.7
show cartoon, occ_591_c2_p0_s0.7 and chain A+D
color palegreen, occ_591_c2_p0_s0.7 and chain A
color lightblue, occ_591_c2_p0_s0.7 and chain D
select hotspot_source, occ_591_c2_p0_s0.7 and ((chain A and resi 498))
select hotspot_target, occ_591_c2_p0_s0.7 and ((chain D and resi 38))
select hotspot_all, occ_591_c2_p0_s0.7 and ((chain A and resi 498) or (chain D and resi 38))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_591_c2_p0_s0.7 and chain A+D
set_name hotspot_all, hotspot_occurrence_591
set_name hotspot_source, hotspot_source_591
set_name hotspot_target, hotspot_target_591
bg_color white
# patternId=0 support=0.7 graphId=389
