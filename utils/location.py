#!/usr/bin/python
# -*- coding: utf-8 -*-
#!/usr/bin/python
# -*- coding: utf-8 -*-

import codecs
import os

debug = []

All_Loc_Prov_dict = {}
Pref_Pinyin_dict = {}
###############################################################################################
########################Town:Province, Town:Prefecture, Town:County################################
twn_prov_dict = {}
twn_pref_dict = {}
twn_cnty_dict = {}
twn1820_pts = codecs.open('data/location/1820/v4_1820_twn_pts_utf.txt', 'r', encoding = 'utf-8')
count = 0
for line in twn1820_pts:
    count += 1
    if count > 1:
        lines = line.strip().split(',')
        twn_prov_dict[lines[3][1:-1]] = lines[18][1:-1]
        twn_pref_dict[lines[3][1:-1]] = lines[20][1:-1]
        Pref_Pinyin_dict[lines[20][1:-1]] = lines[19][1:-1]
        
        twn_cnty_dict[lines[3][1:-1]] = lines[22][1:-1]
        All_Loc_Prov_dict[lines[3][1:-1]] = lines[18][1:-1]
        #print (lines)
#print(twn_prov_dict)

#twn_prov_dict_1911 = {}
#twn_pref_dict_1911 = {}
#twn_cnty_dict_1911 = {}
twn1911_pts = codecs.open('data/location/1911/v4_1911_twn_pts_utf.txt', 'r', encoding = 'utf-8')
count = 0
for line in twn1911_pts:
    count += 1
    if count > 1:
        lines = line.strip().split(',')
        #print (lines)
        #twn_prov_dict_1911[lines[2]] = lines[17]
        #twn_pref_dict_1911[lines[2]] = lines[19]
        #twn_cnty_dict_1911[lines[2]] = lines[21]
        twn_prov_dict[lines[2]] = lines[17]
        twn_pref_dict[lines[2]] = lines[19]
        Pref_Pinyin_dict[lines[19]] = lines[18]
        twn_cnty_dict[lines[2]] = lines[21]
        All_Loc_Prov_dict[lines[2]] = lines[17]
#print (twn_prov_dict_1911)
#print (twn_prov_dict_1911)
####################################################################################################


###############################################################################################
########################County:Province, County:Prefecture################################
cnty_prov_dict = {}
cnty_pref_dict = {}
cnty1820_pts = codecs.open('data/location/1820/v4_1820_cnty_pts_utf.txt', 'r', encoding = 'utf-8')
count = 0
for line in cnty1820_pts:
    count += 1
    if count > 1:
        lines = line.strip().split(',')
        if len(lines[3]) > 2 and '政和' not in lines[3] and '归顺' not in lines[3]:
            cnty_prov_dict[lines[3][:-1]] = lines[18]
            cnty_pref_dict[lines[3][:-1]] = lines[20]
            Pref_Pinyin_dict[lines[20]] = lines[19]
            All_Loc_Prov_dict[lines[3][:-1]] = lines[18] 
        elif lines[3] == '夺营':
            cnty_pref_dict['夺营宗'] = lines[20]
            Pref_Pinyin_dict['夺营宗'] = lines[19]
            cnty_prov_dict['夺营宗'] = lines[18]
        else:
            cnty_prov_dict[lines[3]] = lines[18]
            cnty_pref_dict[lines[3]] = lines[20]
            Pref_Pinyin_dict[lines[20]] = lines[19]
            All_Loc_Prov_dict[lines[3]] = lines[18]
        #print (lines)

#cnty_prov_dict_1911 = {}
#cnty_pref_dict_1911 = {}
cnty1911_png = codecs.open('data/location/1911/v4_1911_cnty_pgn_utf.txt', 'r', encoding = 'utf-8')
cnty1911_pts = codecs.open('data/location/1911/v4_1911_cnty_pts_utf.txt', 'r', encoding = 'utf-8')
count = 0
for line in cnty1911_png:
    count += 1
    if count > 1:
        lines = line.strip().split(',')
        #print (lines)
        #cnty_prov_dict_1911[lines[3]] = lines[16]
        #cnty_pref_dict_1911[lines[3]] = lines[18]
        if len(lines[3]) > 2 and '政和' not in lines[3] and '归顺' not in lines[3]:
            cnty_prov_dict[lines[3][:-1]] = lines[16]
            cnty_pref_dict[lines[3][:-1]] = lines[18]
            Pref_Pinyin_dict[lines[18]] = lines[17]
            All_Loc_Prov_dict[lines[3][:-1]] = lines[16]
        else:
            cnty_prov_dict[lines[3]] = lines[16]
            cnty_pref_dict[lines[3]] = lines[18]
            Pref_Pinyin_dict[lines[18]] = lines[17]
            All_Loc_Prov_dict[lines[3]] = lines[16]
        
count = 0
for line in cnty1911_pts:
    count += 1
    if count > 1:
        lines = line.strip().split(',')
        #cnty_prov_dict_1911[lines[2]] = lines[17]
        #cnty_pref_dict_1911[lines[2]] = lines[19]
        if len(lines[2]) > 2 and '政和' not in lines[2] and '归顺' not in lines[2]:
            cnty_prov_dict[lines[2][:-1]] = lines[17]
            cnty_pref_dict[lines[2][:-1]] = lines[19]
            Pref_Pinyin_dict[lines[19]] = lines[18]
            All_Loc_Prov_dict[lines[2][:-1]] = lines[17]
        else:
            cnty_prov_dict[lines[2]] = lines[17]
            cnty_pref_dict[lines[2]] = lines[19]
            Pref_Pinyin_dict[lines[19]] = lines[18]
            All_Loc_Prov_dict[lines[2]] = lines[17]
        #print (lines)
#print(cnty_prov_dict)
#print (cnty_prov_dict_1911)
####################################################################################################

###############################################################################################
########################Prefecture:Province################################
pref_prov_dict = {}
pref_pref_dict = {}
pref_pinyin_dict = {}
pref1820_png = codecs.open('data/location/1820/v4_1820_pref_pgn_utf.txt', 'r', encoding = 'utf-8')
pref1820_pts = codecs.open('data/location/1820/v4_1820_pref_pts_utf.txt', 'r', encoding = 'utf-8')
count = 0
for line in pref1820_png:
    count += 1
    if count > 1:
        lines = line.strip().split(',')
        #print (lines)
        pref_pinyin_dict[lines[3]] = lines[2]
        if len(lines[3]) > 2 and '政和' not in lines[3] and '归顺' not in lines[3] and '乾州' not in lines[3] and '镇沅' not in lines[3] and '宁远' not in lines[3] and '武定' not in lines[3] and '太平' not in lines[3] and '平定' not in lines[3]:
            #pref_prov_dict[lines[3][:-1]] = lines[16]
            pref_prov_dict[lines[3]] = lines[16]
            pref_pref_dict[lines[3][:-1]] = lines[3]
            Pref_Pinyin_dict[lines[3]] = lines[2]
            All_Loc_Prov_dict[lines[3][:-1]] =  lines[16]
        else:
            pref_prov_dict[lines[3]] = lines[16]
            pref_pref_dict[lines[3]] = lines[3]
            Pref_Pinyin_dict[lines[3]] = lines[2]
            All_Loc_Prov_dict[lines[3]] =  lines[16]
        
count = 0
for line in pref1820_pts:
    count += 1
    if count > 1:
        lines = line.strip().split(',')
        pref_pinyin_dict[lines[3]] = lines[2]
        if len(lines[3]) > 2 and '政和' not in lines[3] and '归顺' not in lines[3] and '乾州' not in lines[3] and '镇沅' not in lines[3] and '宁远' not in lines[3] and '武定' not in lines[3] and '太平' not in lines[3] and '平定' not in lines[3]:
            #pref_prov_dict[lines[3][:-1]] = lines[18]
            pref_prov_dict[lines[3]] = lines[18]
            pref_pref_dict[lines[3][:-1]] = lines[3]
            Pref_Pinyin_dict[lines[3]] = lines[2]
            All_Loc_Prov_dict[lines[3][:-1]] =  lines[18]
        else:
            pref_prov_dict[lines[3]] = lines[18]
            pref_pref_dict[lines[3]] = lines[3]
            Pref_Pinyin_dict[lines[3]] = lines[2]
            All_Loc_Prov_dict[lines[3]] =  lines[18]
        #print (lines)

#pref_prov_dict_1911 = {}
pref1911_png = codecs.open('data/location/1911/v4_1911_pref_pgn_utf.txt', 'r', encoding = 'utf-8')
pref1911_pts = codecs.open('data/location/1911/v4_1911_pref_pts_utf.txt', 'r', encoding = 'utf-8')
count = 0
for line in pref1911_png:
    count += 1
    if count > 1:
        lines = line.strip().split(',')
        #print (lines)
        #pref_prov_dict_1911[lines[3]] = lines[16]
        pref_pinyin_dict[lines[3]] = lines[2]
        if len(lines[3]) > 2 and '政和' not in lines[3] and '归顺' not in lines[3] and '乾州' not in lines[3] and '镇沅' not in lines[3] and '宁远' not in lines[3] and '武定' not in lines[3] and '太平' not in lines[3] and '平定' not in lines[3]:
            #pref_prov_dict[lines[3][:-1]] = lines[16]
            pref_prov_dict[lines[3]] = lines[16]
            pref_pref_dict[lines[3][:-1]] = lines[3]
            Pref_Pinyin_dict[lines[3]] = lines[2]
            All_Loc_Prov_dict[lines[3][:-1]] = lines[16]
        else:
            pref_prov_dict[lines[3]] = lines[16]
            pref_pref_dict[lines[3]] = lines[3]
            Pref_Pinyin_dict[lines[3]] = lines[2]
            All_Loc_Prov_dict[lines[3]] = lines[16]
        
count = 0
for line in pref1911_pts:
    count += 1
    if count > 1:
        lines = line.strip().split(',')
        pref_pinyin_dict[lines[3]] = lines[2]
        #pref_prov_dict_1911[lines[3]] = lines[18]
        if len(lines[3]) > 2 and '政和' not in lines[3] and '归顺' not in lines[3] and '乾州' not in lines[3] and '镇沅' not in lines[3] and '宁远' not in lines[3] and '武定' not in lines[3] and '太平' not in lines[3] and '平定' not in lines[3]:
            #pref_prov_dict[lines[3][:-1]] = lines[18]
            pref_prov_dict[lines[3]] = lines[18]
            pref_pref_dict[lines[3][:-1]] = lines[3]
            Pref_Pinyin_dict[lines[3]] = lines[2]
            All_Loc_Prov_dict[lines[3][:-1]] = lines[18]
        else:
            pref_prov_dict[lines[3]] = lines[18]
            pref_pref_dict[lines[3]] = lines[3]
            Pref_Pinyin_dict[lines[3]] = lines[2]
            All_Loc_Prov_dict[lines[3]] = lines[18]
        #print (lines)
#print (pref_prov_dict)
#print (pref_prov_dict_1911)
####################################################################################################
##################Province:Province######################
prov_prov_dict = {}
prov1820_png = codecs.open('data/location/1820/v4_1820_prov_pgn_utf.txt', 'r', encoding = 'utf-8')
prov1820_pts = codecs.open('data/location/1820/v4_1820_prov_pts_utf.txt', 'r', encoding = 'utf-8')
count = 0
for line in prov1820_png:
    count += 1
    if count > 1:
        lines = line.strip().split(',')
        #print (lines)
        prov_prov_dict[lines[3]] = lines[3]
        All_Loc_Prov_dict[lines[3]] = lines[3]
        
count = 0
for line in prov1820_pts:
    count += 1
    if count > 1:
        lines = line.strip().split(',')
        prov_prov_dict[lines[3][1:-1]] = lines[3][1:-1]
        All_Loc_Prov_dict[lines[3][1:-1]] = lines[3][1:-1]
        #print (lines)

#prov_prov_dict_1911 = {}
prov1911_png = codecs.open('data/location/1911/v4_1911_prov_pgn_utf.txt', 'r', encoding = 'utf-8')
prov1911_pts = codecs.open('data/location/1911/v4_1911_prov_pts_utf.txt', 'r', encoding = 'utf-8')
count = 0
for line in prov1911_png:
    count += 1
    if count > 1:
        lines = line.strip().split(',')
        #print (lines)
        #prov_prov_dict_1911[lines[3]] = lines[3]
        prov_prov_dict[lines[3]] = lines[3]
        All_Loc_Prov_dict[lines[3]] = lines[3]
        
count = 0
for line in prov1911_pts:
    count += 1
    if count > 1:
        lines = line.strip().split(',')
        #prov_prov_dict_1911[lines[3]] = lines[3]
        prov_prov_dict[lines[3]] = lines[3]
        prov_prov_dict['台湾'] = '台湾'
        All_Loc_Prov_dict[lines[3]] = lines[3]
        

#category_list = ['Peasants', 'Jieshe', 'Wuzhuang']
Pref_Pinyin_dict['N/A'] = 'N/A' 

import codecs
import os

All_Loc_Prov_dict = {}
Pref_Pinyin_dict = {}
###############################################################################################
########################Town:Province, Town:Prefecture, Town:County################################
twn_prov_dict = {}
twn_pref_dict = {}
twn_cnty_dict = {}
twn1820_pts = codecs.open('data/location/1820/v4_1820_twn_pts_utf.txt', 'r', encoding = 'utf-8')
count = 0
for line in twn1820_pts:
    count += 1
    if count > 1:
        lines = line.strip().split(',')
        twn_prov_dict[lines[3][1:-1]] = lines[18][1:-1]
        twn_pref_dict[lines[3][1:-1]] = lines[20][1:-1]
        Pref_Pinyin_dict[lines[20][1:-1]] = lines[19][1:-1]
        
        twn_cnty_dict[lines[3][1:-1]] = lines[22][1:-1]
        All_Loc_Prov_dict[lines[3][1:-1]] = lines[18][1:-1]
        #print (lines)
#print(twn_prov_dict)

#twn_prov_dict_1911 = {}
#twn_pref_dict_1911 = {}
#twn_cnty_dict_1911 = {}
twn1911_pts = codecs.open('data/location/1911/v4_1911_twn_pts_utf.txt', 'r', encoding = 'utf-8')
count = 0
for line in twn1911_pts:
    count += 1
    if count > 1:
        lines = line.strip().split(',')
        #print (lines)
        #twn_prov_dict_1911[lines[2]] = lines[17]
        #twn_pref_dict_1911[lines[2]] = lines[19]
        #twn_cnty_dict_1911[lines[2]] = lines[21]
        twn_prov_dict[lines[2]] = lines[17]
        twn_pref_dict[lines[2]] = lines[19]
        Pref_Pinyin_dict[lines[19]] = lines[18]
        twn_cnty_dict[lines[2]] = lines[21]
        All_Loc_Prov_dict[lines[2]] = lines[17]
#print (twn_prov_dict_1911)
#print (twn_prov_dict_1911)
####################################################################################################


###############################################################################################
########################County:Province, County:Prefecture################################
cnty_prov_dict = {}
cnty_pref_dict = {}
cnty1820_pts = codecs.open('data/location/1820/v4_1820_cnty_pts_utf.txt', 'r', encoding = 'utf-8')
count = 0
for line in cnty1820_pts:
    count += 1
    if count > 1:
        lines = line.strip().split(',')
        if len(lines[3]) > 2 and '政和' not in lines[3] and '归顺' not in lines[3]:
            cnty_prov_dict[lines[3][:-1]] = lines[18]
            cnty_pref_dict[lines[3][:-1]] = lines[20]
            Pref_Pinyin_dict[lines[20]] = lines[19]
            All_Loc_Prov_dict[lines[3][:-1]] = lines[18] 
        elif lines[3] == '夺营':
            cnty_pref_dict['夺营宗'] = lines[20]
            Pref_Pinyin_dict['夺营宗'] = lines[19]
            cnty_prov_dict['夺营宗'] = lines[18]
        else:
            cnty_prov_dict[lines[3]] = lines[18]
            cnty_pref_dict[lines[3]] = lines[20]
            Pref_Pinyin_dict[lines[20]] = lines[19]
            All_Loc_Prov_dict[lines[3]] = lines[18]
        #print (lines)

#cnty_prov_dict_1911 = {}
#cnty_pref_dict_1911 = {}
cnty1911_png = codecs.open('data/location/1911/v4_1911_cnty_pgn_utf.txt', 'r', encoding = 'utf-8')
cnty1911_pts = codecs.open('data/location/1911/v4_1911_cnty_pts_utf.txt', 'r', encoding = 'utf-8')
count = 0
for line in cnty1911_png:
    count += 1
    if count > 1:
        lines = line.strip().split(',')
        #print (lines)
        #cnty_prov_dict_1911[lines[3]] = lines[16]
        #cnty_pref_dict_1911[lines[3]] = lines[18]
        if len(lines[3]) > 2 and '政和' not in lines[3] and '归顺' not in lines[3]:
            cnty_prov_dict[lines[3][:-1]] = lines[16]
            cnty_pref_dict[lines[3][:-1]] = lines[18]
            Pref_Pinyin_dict[lines[18]] = lines[17]
            All_Loc_Prov_dict[lines[3][:-1]] = lines[16]
        else:
            cnty_prov_dict[lines[3]] = lines[16]
            cnty_pref_dict[lines[3]] = lines[18]
            Pref_Pinyin_dict[lines[18]] = lines[17]
            All_Loc_Prov_dict[lines[3]] = lines[16]
        
count = 0
for line in cnty1911_pts:
    count += 1
    if count > 1:
        lines = line.strip().split(',')
        #cnty_prov_dict_1911[lines[2]] = lines[17]
        #cnty_pref_dict_1911[lines[2]] = lines[19]
        if len(lines[2]) > 2 and '政和' not in lines[2] and '归顺' not in lines[2]:
            cnty_prov_dict[lines[2][:-1]] = lines[17]
            cnty_pref_dict[lines[2][:-1]] = lines[19]
            Pref_Pinyin_dict[lines[19]] = lines[18]
            All_Loc_Prov_dict[lines[2][:-1]] = lines[17]
        else:
            cnty_prov_dict[lines[2]] = lines[17]
            cnty_pref_dict[lines[2]] = lines[19]
            Pref_Pinyin_dict[lines[19]] = lines[18]
            All_Loc_Prov_dict[lines[2]] = lines[17]
        #print (lines)
#print(cnty_prov_dict)
#print (cnty_prov_dict_1911)
####################################################################################################

###############################################################################################
########################Prefecture:Province################################
pref_prov_dict = {}
pref_pref_dict = {}
pref_pinyin_dict = {}
pref1820_png = codecs.open('data/location/1820/v4_1820_pref_pgn_utf.txt', 'r', encoding = 'utf-8')
pref1820_pts = codecs.open('data/location/1820/v4_1820_pref_pts_utf.txt', 'r', encoding = 'utf-8')
count = 0
for line in pref1820_png:
    count += 1
    if count > 1:
        lines = line.strip().split(',')
        #print (lines)
        pref_pinyin_dict[lines[3]] = lines[2]
        if len(lines[3]) > 2 and '政和' not in lines[3] and '归顺' not in lines[3] and '乾州' not in lines[3] and '镇沅' not in lines[3] and '宁远' not in lines[3] and '武定' not in lines[3] and '太平' not in lines[3] and '平定' not in lines[3]:
            #pref_prov_dict[lines[3][:-1]] = lines[16]
            pref_prov_dict[lines[3]] = lines[16]
            pref_pref_dict[lines[3][:-1]] = lines[3]
            Pref_Pinyin_dict[lines[3]] = lines[2]
            All_Loc_Prov_dict[lines[3][:-1]] =  lines[16]
        else:
            pref_prov_dict[lines[3]] = lines[16]
            pref_pref_dict[lines[3]] = lines[3]
            Pref_Pinyin_dict[lines[3]] = lines[2]
            All_Loc_Prov_dict[lines[3]] =  lines[16]
        
count = 0
for line in pref1820_pts:
    count += 1
    if count > 1:
        lines = line.strip().split(',')
        pref_pinyin_dict[lines[3]] = lines[2]
        if len(lines[3]) > 2 and '政和' not in lines[3] and '归顺' not in lines[3] and '乾州' not in lines[3] and '镇沅' not in lines[3] and '宁远' not in lines[3] and '武定' not in lines[3] and '太平' not in lines[3] and '平定' not in lines[3]:
            #pref_prov_dict[lines[3][:-1]] = lines[18]
            pref_prov_dict[lines[3]] = lines[18]
            pref_pref_dict[lines[3][:-1]] = lines[3]
            Pref_Pinyin_dict[lines[3]] = lines[2]
            All_Loc_Prov_dict[lines[3][:-1]] =  lines[18]
        else:
            pref_prov_dict[lines[3]] = lines[18]
            pref_pref_dict[lines[3]] = lines[3]
            Pref_Pinyin_dict[lines[3]] = lines[2]
            All_Loc_Prov_dict[lines[3]] =  lines[18]
        #print (lines)

#pref_prov_dict_1911 = {}
pref1911_png = codecs.open('data/location/1911/v4_1911_pref_pgn_utf.txt', 'r', encoding = 'utf-8')
pref1911_pts = codecs.open('data/location/1911/v4_1911_pref_pts_utf.txt', 'r', encoding = 'utf-8')
count = 0
for line in pref1911_png:
    count += 1
    if count > 1:
        lines = line.strip().split(',')
        #print (lines)
        #pref_prov_dict_1911[lines[3]] = lines[16]
        pref_pinyin_dict[lines[3]] = lines[2]
        if len(lines[3]) > 2 and '政和' not in lines[3] and '归顺' not in lines[3] and '乾州' not in lines[3] and '镇沅' not in lines[3] and '宁远' not in lines[3] and '武定' not in lines[3] and '太平' not in lines[3] and '平定' not in lines[3]:
            #pref_prov_dict[lines[3][:-1]] = lines[16]
            pref_prov_dict[lines[3]] = lines[16]
            pref_pref_dict[lines[3][:-1]] = lines[3]
            Pref_Pinyin_dict[lines[3]] = lines[2]
            All_Loc_Prov_dict[lines[3][:-1]] = lines[16]
        else:
            pref_prov_dict[lines[3]] = lines[16]
            pref_pref_dict[lines[3]] = lines[3]
            Pref_Pinyin_dict[lines[3]] = lines[2]
            All_Loc_Prov_dict[lines[3]] = lines[16]
        
count = 0
for line in pref1911_pts:
    count += 1
    if count > 1:
        lines = line.strip().split(',')
        pref_pinyin_dict[lines[3]] = lines[2]
        #pref_prov_dict_1911[lines[3]] = lines[18]
        if len(lines[3]) > 2 and '政和' not in lines[3] and '归顺' not in lines[3] and '乾州' not in lines[3] and '镇沅' not in lines[3] and '宁远' not in lines[3] and '武定' not in lines[3] and '太平' not in lines[3] and '平定' not in lines[3]:
            #pref_prov_dict[lines[3][:-1]] = lines[18]
            pref_prov_dict[lines[3]] = lines[18]
            pref_pref_dict[lines[3][:-1]] = lines[3]
            Pref_Pinyin_dict[lines[3]] = lines[2]
            All_Loc_Prov_dict[lines[3][:-1]] = lines[18]
        else:
            pref_prov_dict[lines[3]] = lines[18]
            pref_pref_dict[lines[3]] = lines[3]
            Pref_Pinyin_dict[lines[3]] = lines[2]
            All_Loc_Prov_dict[lines[3]] = lines[18]
        #print (lines)
#print (pref_prov_dict)
#print (pref_prov_dict_1911)
####################################################################################################
##################Province:Province######################
prov_prov_dict = {}
prov1820_png = codecs.open('data/location/1820/v4_1820_prov_pgn_utf.txt', 'r', encoding = 'utf-8')
prov1820_pts = codecs.open('data/location/1820/v4_1820_prov_pts_utf.txt', 'r', encoding = 'utf-8')
count = 0
for line in prov1820_png:
    count += 1
    if count > 1:
        lines = line.strip().split(',')
        #print (lines)
        prov_prov_dict[lines[3]] = lines[3]
        All_Loc_Prov_dict[lines[3]] = lines[3]
        
count = 0
for line in prov1820_pts:
    count += 1
    if count > 1:
        lines = line.strip().split(',')
        prov_prov_dict[lines[3][1:-1]] = lines[3][1:-1]
        All_Loc_Prov_dict[lines[3][1:-1]] = lines[3][1:-1]
        #print (lines)

#prov_prov_dict_1911 = {}
prov1911_png = codecs.open('data/location/1911/v4_1911_prov_pgn_utf.txt', 'r', encoding = 'utf-8')
prov1911_pts = codecs.open('data/location/1911/v4_1911_prov_pts_utf.txt', 'r', encoding = 'utf-8')
count = 0
for line in prov1911_png:
    count += 1
    if count > 1:
        lines = line.strip().split(',')
        #print (lines)
        #prov_prov_dict_1911[lines[3]] = lines[3]
        prov_prov_dict[lines[3]] = lines[3]
        All_Loc_Prov_dict[lines[3]] = lines[3]
        
count = 0
for line in prov1911_pts:
    count += 1
    if count > 1:
        lines = line.strip().split(',')
        #prov_prov_dict_1911[lines[3]] = lines[3]
        prov_prov_dict[lines[3]] = lines[3]
        prov_prov_dict['台湾'] = '台湾'
        All_Loc_Prov_dict[lines[3]] = lines[3]



'''############To check what falls into '河南府' etc.############
#河南府，顺天府，云南府，伊犁，广西州，靖州，台湾府
Taiwanfu = []
for item in pref_pref_dict.keys():
    if pref_pref_dict[item] == '台湾府':
        if item not in Taiwanfu:
            Taiwanfu.append(item)
for item in cnty_pref_dict.keys():
    if cnty_pref_dict[item] == '台湾府':
        if item not in Taiwanfu:
            Taiwanfu.append(item)
for item in twn_pref_dict.keys():
    if twn_pref_dict[item] == '台湾府':
        if item not in Taiwanfu:
            Taiwanfu.append(item)
#print (len(set(Henanfu))) # = 180
#print (len(set(Shuntianfu))) # = 615
#print (len(Chenzhoufu)) # = 138  #Chenzhoufu is used as an compare-example. There are only 157 riots in Chenzhoufu. 
#print (len(Taiwanfu)) # = 52
#print (Taiwanfu)

Henanfu = []
for item in pref_pref_dict.keys():
    if pref_pref_dict[item] == '河南府':
        if item not in Henanfu:
            Henanfu.append(item)
for item in cnty_pref_dict.keys():
    if cnty_pref_dict[item] == '河南府':
        if item not in Henanfu:
            Henanfu.append(item)
for item in twn_pref_dict.keys():
    if twn_pref_dict[item] == '河南府':
        if item not in Henanfu:
            Henanfu.append(item)
print (Henanfu)

prefectures = []
prefectures_more = []
prefectures_short = []
for item in pref_pref_dict.values():
    if item != ' ' and item != '':
        prefectures.append(item)
        prefectures_more.append(item)
        prefectures_short.append(item[:2])
prefectures.append('N/A')
#pref_prov_dict['N/A'] = ' '
print (prefectures)
for item in cnty_pref_dict.values():
    if item != ' ' and item != '':
        prefectures_more.append(item)
        
for item in twn_pref_dict.values():
    if item != ' ' and item != '':
        prefectures_more.append(item)

no_short_list = []
no_long_list = []
for item in set(prefectures_more):
    if item[:2] not in prefectures_short:
        no_short_list.append(item)
    if item not in prefectures:
        no_long_list.append(item)
        
#print (len(no_short_list))
#print (len(no_long_list))
#print (no_short_list)

#print (pref_pref_dict)

'''
prefectures = ['陈州府', '商州', '三音诺颜部', '临安府', '乾州', '泗州', '和州', '九江府', '都匀府', '南安府', '辽州', '乌噜木齐', '广平府', '东川府', '阿克苏', '固原州', '呼伦贝尔副都统辖区', '吉林', '达里冈爱牧场', '海州', '黄州府', '郴州', '赵州', '扬州府', '化平川厅', '肃州', '大定府', '永昌府', '定海厅', '汝宁府', '布特哈', '萨拉齐厅', '成都府', '衡州府', '平越州', '顺庆府', '乌兰察布盟', '曲靖府', '河南府', '汉中府', '龙安府', '眉州', '千里石塘', '南宁府', '汾州府', '昭通府', '杭州府', '松桃厅', '陶林厅', '永顺府', '湖州府', '科布多', '赣州府', '墨尔根副都统辖区', '思州府', '庐州府', '滁州', '镇雄州', '许州', '忻州', '太仓州', '东昌府', '齐齐哈尔', '奉天府', '遵化州', '青海', '通州', '霍州', '宁远府', '顺天府', '沁州', '定州', '郁林州', '伊克昭盟', '晃州厅', '淮安府', '重庆府', '徐州府', '万里长沙', '河间府', '广信府', '德安府', '南阳府', '沅州府', '和阗', '哲里木盟', '饶州府', '云南府', '高州府', '资州', '泽州府', '镇远府', '彰德府', '墨尔根', '唐努乌梁海', '宁古塔', '甘州府', '金华府', '沂州府', '邵武府', '酉阳州', '仁怀厅', '袁州府', '中俄尼布楚条约待议地区', '普安厅', '江宁府', '宁波府', '浔州府', '雅州府', '伊犁', '永平府', '武定州', '归化绥远城', '澧州', '南康府', '承德府', '白都纳副都统辖区', '大同府', '阳江厅', '大名府', '解州', '郧阳府', '太平府', '平乐府', '白都讷', '宁国府', '顺德府', '澄江府', '昭乌达盟', '处州府', '龙岩州', '莱州府', '养息牧场', '武昌府', '建昌府', '兴安府', '广南府', '乾州厅', '吉安府', '广西州', '西宁府', '靖州', '天津府', '五原厅', '长沙府', '齐齐哈尔副都统辖区', '泗城府', '绛州', '济宁州', '延安府', '苏州府', '易州', '库尔喀喇乌苏', '六安州', '台州府', '陕州', '百色厅', '永春州', '归德府', '福宁府', '杂谷厅', '呼伦布雨尔', '颍州府', '雷州府', '廉州府', '济南府', '广州府', '阶州', '西藏', '迪化州', '延平府', '福州府', '和林格尔厅', '庆远府', '永北厅', '赤溪厅', '韶州府', '南昌府', '宣化府', '嘉定府', '汝州', '同州府', '琼州府', '荆州府', '代州', '喀喇沙尔', '广德州', '黎平府', '吐鲁番', '赤峰州', '古城', '朔平府', '镇边厅', '开化府', '隰州', '凉州府', '吉林副都统辖区', '三姓副都统辖区', '黑龙江副都统辖区', '台湾府', '遵义府', '清水河厅', '衢州府', '安陆府', '土谢图汗部', '哈密', '连州', '丽江府', '普洱府', '阿拉善厄鲁特旗', '兖州府', '兴化府', '永州府', '庆阳府', '懋功厅', '绵州', '喀喇乌苏', '淅川厅', '乌什', '柳州府', '常州府', '惠州府', '常德府', '镇西府', '归化城厅', '丰镇厅', '平定州', '凤阳府', '肇庆府', '库车', '三姓', '南澳厅', '梧州府', '开封府', '连山厅', '光州', '石阡府', '铜仁府', '罗定州', '榆林府', '景东厅', '潮州府', '多伦诺尔厅', '大凌河牧场', '忠州', '兴和厅', '瑞州府', '温州府', '东沙', '秦州', '宜昌府', '巴里坤', '黑龙江城', '叙州府', '曹州府', '蒲州府', '青州府', '额济纳土尔扈特旗', '泉州府', '汀州府', '顺宁府', '安西州', '佛冈厅', '邛州', '思恩府', '保德州', '保定府', '太原府', '锡林郭勒盟', '登州府', '独石口厅', '泾州', '巩昌府', '贵阳府', '宁都州', '卫辉府', '怀庆府', '扎萨克图汗部', '泸州', '泰安府', '桂林府', '上思厅', '宝庆府', '兰州府', '太平厅', '正定府', '宁古塔副都统辖区', '建宁府', '潞安府', '腾越厅', '思南府', '喇萨', '宁夏府', '施南府', '武川厅', '阿勒楚喀副都统辖区', '郑州', '镇安府', '平阳府', '喀什噶尔', '钦州', '鄜州', '车臣汗部', '兴义府', '汉阳府', '潼川府', '镇沅厅', '东胜厅', '桂阳州', '绥定府', '冀州', '澂江府', '安顺府', '嘉应州', '库伦', '凤翔府', '镇江府', '武定府', '阿勒楚喀', '蒙化厅', '池州府', '绥远城厅', '海门厅', '松江府', '归绥六厅', '归顺州', '邠州', '乌里雅苏台', '荆门州', '石砫厅', '宁武府', '南雄州', '嘉兴府', '宁远厅', '镇沅州', '西安府', '凤凰厅', '保宁府', '绍兴府', '绥德州', '元江州', '叙永厅', '平凉府', '安庆府', '大理府', '夔州府', '日喀则', '岳州府', '临清州', '漳州府', '襄阳府', '深州', '辰州府', '胶州', '朝阳府', '托克托厅', '锦州府', '临江府', '抚州府', '张家口厅', '松潘厅', '楚雄府', '茂州', '口北三厅', '徽州府', '叶尔羌', '永绥厅', '塔尔巴哈台', '严州府', 'N/A']
Pref_Pinyin_dict['N/A'] = 'N/A'       

# delete some weird towns
for loc in ['河南']:
    del twn_pref_dict[loc]

def TimeS(inpt, reign):
    f = codecs.open(inpt, 'r', encoding = 'utf-8')
    timelist = []
    for line in f:
        if reign in line:
            lines = line.strip().split()
            indx = lines[2].index('年')
            if lines[2][0] == '○':
                year = lines[2][1:indx+1]
            else:
                year = lines[2][:indx+1]
            timelist.append(year)
    return timelist
    

def InOrNot(entry, loc_names, ins_list):
    for nm in loc_names.keys():
        if len(nm) == 1:
            continue
        if nm in ['同心', '王令', '通道', '军营', '会同', '江北', '世忠', '江南', '江口', '前所', '后路', '大获', "山东","黑龙江","福建","内蒙古","乌里雅苏台","山西","浙江","广西","西藏","甘肃","云南","四川","湖南","江 西","江苏","直 隶","盛京","新疆","陕西","贵州","吉林","河南","青海","None","湖北","安徽","广东", '顺天', '奉天', '天津']:
            continue
        if nm in entry:
            debug.append(nm)
            ins_list.append(nm)
    return ins_list
      
def LocationIdentify(line):
    pref_name_list = []
    # level_list = [pref_pref_dict, cnty_pref_dict, twn_pref_dict]
    level_list = [pref_pref_dict, cnty_pref_dict]
    #ct = 0
    for level in level_list:
        ins_list = []
        #ct += 1
        #print (ct)
        ins_list = InOrNot(line, level, ins_list)
        if len(ins_list) > 0:
            #print (ins_list)
            for ins in ins_list:
                #print(level[ins])
                if level[ins] == 'Mingying Tuzhou':
                    level[ins] = '太平府'
                if level[ins] == '柳柳州府':
                    level[ins] = '柳州府'
                if level[ins] == '雅安府':
                    level[ins] = '雅州府'
                if level[ins] == '宁武县':
                    level[ins] = '宁武府'
                if level[ins] == '察哈尔':
                    level[ins] = '陶林厅'
                if level[ins] == '呼兰府':
                    level[ins] = '齐齐哈尔副都统辖区'
                if level[ins] == '绥化府':
                    level[ins] = '平越州'
                if level[ins] == '吉林副都统':
                    level[ins] = '吉林副都统辖区'
                if level[ins] == '通州直隶州':
                    level[ins] = '通州'
                if level[ins] == '绥德直隶州':
                    level[ins] = '绥德州'
                if level[ins] == '永春直隶州':
                    level[ins] = '永春州'
                if level[ins] == '资州直隶州':
                    level[ins] = '资州'
                if level[ins] == '海门直隶厅':
                    level[ins] = '海门厅'
                if level[ins] == '茂州直隶州':
                    level[ins] = '茂州'
                if level[ins] == '泾州直隶州':
                    level[ins] = '泾州'
                if level[ins] == '归化绥远城厅': 
                    level[ins] = '归化绥远城'
                if level[ins] == '归顺直隶州':
                    level[ins] = '归顺州'
                if level[ins] == '商州直隶州':
                    level[ins] = '商州'
                if level[ins] == '太仓直隶州':
                    level[ins] = '太仓州'
                if level[ins] == '懋功直隶厅':
                    level[ins] = '懋功厅'
                if level[ins] == '邛州直隶州':
                    level[ins] = '邛州'
                if level[ins] == '海州直隶州':
                    level[ins] = '海州'
                if level[ins] == '镇沅州':
                    level[ins] = '镇沅州'
                if level[ins] == '龙岩直隶州':
                    level[ins] = '龙岩州'
                if level[ins] == '忠州直隶州':
                    level[ins] = '忠州'
                if level[ins] == '松桃直隶厅':
                    level[ins] = '松桃厅'
                if level[ins] == '墨尔根副都统':
                    level[ins] = '墨尔根副都统辖区'
                if level[ins] == '阿勒楚喀副都统':
                    level[ins] = '阿勒楚喀副都统辖区'
                if level[ins] == '石砫直隶厅': 
                    level[ins] = '石砫厅'
                if level[ins] == '泸州直隶州':
                    level[ins] = '泸州'
                if level[ins] == '宜昌县':
                    level[ins] = '宜昌府'
                if level[ins] == '固原直隶州':
                    level[ins] = '固原州'
                if level[ins] == '定州直隶州':
                    level[ins] = '定州'
                if level[ins] == '易州直隶州':
                    level[ins] = '易州'
                if level[ins] == '白都讷副都统':
                    level[ins] =  '白都讷'
                if level[ins] == '雅安府':
                    level[ins] = '雅州府'
                if level[ins] == '平越直隶州':
                    level[ins] = '平越州'
                if level[ins] == '邠州直隶州':
                    level[ins] = '邠州'
                if level[ins] == '重庆市':
                    level[ins] = '重庆府'
                if level[ins] == '宁远府': 
                    level[ins] = '宁远府'
                if level[ins] == '百色直隶厅':
                    level[ins] = '百色厅'
                if level[ins] == '遵化直隶州':
                    level[ins] = '遵化州'
                if level[ins] == '冀州直隶州': 
                    level[ins] = '冀州'
                if level[ins] == '绵州直隶州':
                    level[ins] = '绵州'
                if level[ins] == '绥化府':
                    level[ins] = '平越州'
                if level[ins] == '秦州直隶州':
                    level[ins] = '秦州'
                if level[ins] == '三姓副都统':
                    level[ins] = '三姓副都统辖区'
                if level[ins] == '沅州府沅州府':
                    level[ins] = '沅州府'
                if level[ins] == '鄜州直隶州':
                    level[ins] = '鄜州'
                if level[ins] == '乾州直隶州':
                    level[ins] = '乾州'
                if level[ins] == '安西直隶州':
                    level[ins] = '安西州'
                if level[ins] == '深州直隶州':
                    level[ins] = '深州'
                if level[ins] == '黑龙江副都统':
                    level[ins] = '黑龙江副都统辖区'
                if level[ins] == '酉阳直隶州':
                    level[ins] = '酉阳州'
                if level[ins] == '上思直隶厅':
                    level[ins] = '上思厅'
                if level[ins] == '武定府':
                    level[ins] = '武定府'
                if level[ins] == '化平川直隶厅':
                    level[ins] = '化平川厅'
                if level[ins] == '赵州直隶州':
                    level[ins] = '赵州'
                if level[ins] == '眉州直隶州':
                    level[ins] = '眉州'
                if level[ins] == '呼伦贝尔总管':
                    level[ins] = '呼伦贝尔副都统辖区'
                if level[ins] == '肃州直隶州':
                    level[ins] = '肃州'
                if level[ins] == '宁武县':
                    level[ins] = '宁武府'
                if level[ins] == '松潘直隶厅':
                    level[ins] = '松潘厅'
                if level[ins] == '宁古塔副都统':
                    level[ins] = '宁古塔副都统辖区'
                if level[ins] == '阶州直隶州':
                    level[ins] = '阶州'
                if level[ins] == '郁林直隶州':
                    level[ins] = '郁林州'
                if level[ins] == '齐齐哈尔副都统':
                    level[ins] = '齐齐哈尔副都统辖区'
                    
                if level[ins] != '' and level[ins] != ' ':
                    pref_name_list.append(level[ins])
                    #print (level[ins])
                
    return list(set(pref_name_list))
