# import glob
# import replicate

# # imagefile_regex = "../data/lrdataset/data/1.tif"
# # filenames = glob.glob(imagefile_regex)
# # for file in filenames:
# output = replicate.run(
# "rmokady/clip_prefix_caption:9a34a6339872a03f45236f114321fb51fc7aa8269d38ae0ce5334969981e4cd8",
# input={
#     "image": "https://replicate.delivery/mgxm/4dc7763a-f234-4a7c-a85f-cb9e05e37cf8/COCO_val2014_000000579664.jpg",
#     "model": "coco",
#     "use_beam_search": False
# }
# )

# print(output)