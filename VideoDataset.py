import os
import pprint

video_base_path = "/Data/xcy_test"
bms_file_base_path = "/Data/Bitstream/ECM11"

video_sequences = {
  "A1": ["Tango2", "FoodMarket4", "Campfire"],
  "A2": ["CatRobot", "DaylightRoad2", "ParkRunning3"],
  "B": ["MarketPlace", "RitualDance", "Cactus", "BasketballDrive", "BQTerrace"],
  "C": ["BasketballDrill", "BQMall", "PartyScene", "RaceHorsesC"], 
  "D": ["BasketballPass", "BlowingBubbles", "BQSquare", "RaceHorses"],
  "E": ["FourPeople", "Johnny", "KristenAndSara"],
  "F": ["BasketballDrillText", "ArenaOfValor", "SlideEditing", "SlideShow"],
  "TGM": ["FlyingGraphic", "Desktop", "Console", "ChineseEditing"]
}

video_width_height_bitdepth = {
  "A1": [(3840, 2160, 10), (3840, 2160, 10), (3840, 2160, 10)],
  "A2": [(3840, 2160, 10), (3840, 2160, 10), (3840, 2160, 10)],
  "B": [(1920, 1080, 10), (1920, 1080, 10), (1920, 1080, 8), (1920, 1080, 8), (1920, 1080, 8)],
  "C": [(832, 480, 8), (832, 480, 8), (832, 480, 8), (832, 480, 8)], 
  "D": [(416, 240, 8), (416, 240, 8), (416, 240, 8), (416, 240, 8)],
  "E": [(1280, 720, 8), (1280, 720, 8), (1280, 720, 8)],
  "F": [(832, 480, 8), (1920, 1080, 8), (1280, 720, 8), (1280, 720, 8)],
  "TGM": [(1920, 1080, 8), (1920, 1080, 8), (1920, 1080, 8), (1920, 1080, 8)]
}

video_file_names = {
  'A1': ['/Data/xcy_test/ClassA1/Tango2_3840x2160_60fps_10bit_420.yuv', '', ''],
  'A2': ['', '', ''],
  'B': ['',
       '/Data/xcy_test/ClassB/RitualDance_1920x1080_60fps_10bit_420.yuv',
       '/Data/xcy_test/ClassB/Cactus_1920x1080_50.yuv',
       '/Data/xcy_test/ClassB/BasketballDrive_1920x1080_50.yuv',
       '/Data/xcy_test/ClassB/BQTerrace_1920x1080_60.yuv'],
  'C': ['/Data/xcy_test/ClassC/BasketballDrill_832x480_50.yuv',
       '/Data/xcy_test/ClassC/BQMall_832x480_60.yuv',
       '/Data/xcy_test/ClassC/PartyScene_832x480_50.yuv',
       '/Data/xcy_test/ClassC/RaceHorses_832x480_30.yuv'],
  'D': ['/Data/xcy_test/ClassD/BasketballPass_416x240_50.yuv',
       '/Data/xcy_test/ClassD/BlowingBubbles_416x240_50.yuv',
       '/Data/xcy_test/ClassD/BQSquare_416x240_60.yuv',
       '/Data/xcy_test/ClassD/RaceHorses_416x240_30.yuv'],
  'E': ['/Data/xcy_test/ClassE/FourPeople_1280x720_60.yuv',
       '/Data/xcy_test/ClassE/Johnny_1280x720_60.yuv',
       '/Data/xcy_test/ClassE/KristenAndSara_1280x720_60.yuv'],
  'F': ['/Data/xcy_test/ClassF/BasketballDrillText_832x480_50.yuv',
       '',
       '/Data/xcy_test/ClassF/SlideEditing_1280x720_30.yuv',
       '/Data/xcy_test/ClassF/SlideShow_1280x720_20.yuv'],
  'TGM': ['',
         '',
         '',
         '/Data/xcy_test/ClassTGM/ChineseEditing_1920x1080_60_8bit_420.yuv']
}

video_decoded_file_names = {}

def find_files_with_string(folder_path, search_string):
    found_files = []
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith('.yuv') and search_string in file_name:
                found_files.append(os.path.join(root, file_name))
    return found_files

def find_stats_files(folder_path, video_sequence, qp):
    found_files = []
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if video_sequence in file_name and qp + '.vtmbmsstats' in file_name:
                found_files.append(os.path.join(root, file_name))
    return found_files

# # get video file names
# for key, values in video_sequences.items():
#   video_file_names[key] = []
#   for seq in values:
#     found = find_files_with_string(video_base_path, seq)
#     if found:
#       print(seq, found[0].replace('\\', '/'))
#       video_file_names[key].append(found[0].replace('\\', '/'))
#     else:
#       print(seq)
#       video_file_names[key].append("")
# pprint.pprint(video_file_names)

# get video decoded file names
# for key, values in video_sequences.items():
#   video_file_names[key] = []
#   for seq in values:
#     found = find_files_with_string(bms_file_base_path, seq)
#     if found:
#       print(seq, found[0].replace('\\', '/'))
#       video_file_names[key].append(found[0].replace('\\', '/'))
#     else:
#       print(seq)
#       video_file_names[key].append("")
# pprint.pprint(video_file_names)
