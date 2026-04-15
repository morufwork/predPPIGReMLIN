load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb8dma.ent", occ_2211_c4_p0_s0.8
hide everything, occ_2211_c4_p0_s0.8
show cartoon, occ_2211_c4_p0_s0.8 and chain A+D
color palegreen, occ_2211_c4_p0_s0.8 and chain A
color lightblue, occ_2211_c4_p0_s0.8 and chain D
select hotspot_source, occ_2211_c4_p0_s0.8 and ((chain A and resi 501))
select hotspot_target, occ_2211_c4_p0_s0.8 and ((chain D and resi 353))
select hotspot_all, occ_2211_c4_p0_s0.8 and ((chain A and resi 501) or (chain D and resi 353))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_2211_c4_p0_s0.8 and chain A+D
set_name hotspot_all, hotspot_occurrence_2211
set_name hotspot_source, hotspot_source_2211
set_name hotspot_target, hotspot_target_2211
bg_color white
# patternId=0 support=0.8 graphId=394
