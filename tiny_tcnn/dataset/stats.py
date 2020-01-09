# # mpeg4 upsampled, training data, only static cameras (Note: Theese metrics still include the num_boxes_mask bug)
# class StatsMpeg4UpsampledStatic():
#     velocities = {
#         "mean": [0.0035801701807530333, 0.0002152514312298825, 0.00021574009192311123, 8.994028635911342e-05],
#         "std": [0.09732869575370293, 0.012557833992140246, 0.01932378335492126, 0.004216800692202473]
#     }
#     motion_vectors = {
#         "mean": [0.0, 0.014817102005829075, 0.0705440781107341], # [0.0, mvs_mean_y, mvs_mean_x]
#         "std": [1.0, 0.060623864350822454,  0.4022695243698158]  # [1.0, mvs_std_y, mvs_std_x]
#     }
#
# # mpeg4 upsampled, training data, full dataset
# class StatsMpeg4UpsampledFull():
#     velocities = {
#         "mean": [-0.018599934971495233, 0.0025663658130240146, 0.002417661236454852, 0.002733427047478466],
#         "std": [0.21269956851634267, 0.036000079453724264, 0.044600952658130015, 0.014932423372624959]
#     }
#     motion_vectors = {
#         "mean": [0.0, 0.139698425141241, -0.16339078281237884],
#         "std": [1.0, 0.9256943895550922,  2.821564673026061]
#     }

######################### NEW DATASET (excludes keyframes) #########################

# mpeg4 upsampled, no keyframes, training data, only static cameras
# codec = mpeg4, mvs_mode = upsampled, static_only = True, exclude_keyframes = True, scales = [1.0]
class StatsMpeg4UpsampledStaticSinglescale():
    velocities = {
        "mean": [0.003505179250665827, 0.00018602565541264086, 0.0002404537348835983, 8.181186571028626e-05],
        "std": [0.09751977443928779, 0.012586229216352347, 0.0193288116599051, 0.0042137634660418125]
    }
    motion_vectors = {
        "mean": [[0.0, 0.016144431789511565, 0.07686348226290671], []],
        "std": [[1.0, 0.06311146108698115, 0.41932218398916854], []]
    }


# mpeg4 upsampled, no keyframes, training data, full dataset
# codec = mpeg4, mvs_mode = upsampled, static_only = False, exclude_keyframes = True, scales = [1.0]
class StatsMpeg4UpsampledFullSinglescale():
    velocities = {
        "mean": [-0.018749966234393636, 0.002650610891189643, 0.0024707428297439603, 0.002743245574765845],
        "std": [0.213048754134384, 0.036284140941397, 0.04492437750683111, 0.014939794555400754]
    }
    motion_vectors = {
        "mean": [[0.0, 0.152116229012106, -0.17859823792827198], []],
        "std": [[1.0, 0.965357249505038, 2.9459254858101116], []]
    }


# h264 upsampled, no keyframes, training data, only static cameras
# codec = h264, mvs_mode = upsampled, static_only = True, exclude_keyframes = True, scales = [1.0]
class StatsH264UpsampledStaticSinglescale():
    velocities = {
        "mean": [0.0035696855836549668, 0.0002081201615972408, 0.0002175107888165752, 8.973345914091878e-05],
        "std": [0.09733716307340097, 0.012554011163439764, 0.019332615926499656, 0.004220630612865972]
    }
    motion_vectors = {
        "mean": [[0.0, 0.012684933064480106, 0.057034346243264596],
                 [0.0, 0.006924102768413737, 0.04841236712117815]],
        "std": [[1.0, 0.0789567813024147, 0.3226359114106121],
                [1.0, 0.03428847744895673, 0.20594691881916388]]
    }


# h264 upsampled, no keyframes, training data, full dataset
# codec = h264, mvs_mode = upsampled, static_only = False, exclude_keyframes = True, scales = [1.0]
class StatsH264UpsampledFullSinglescale():
    velocities = {
        "mean": [-0.01869362938217712, 0.002564289825545628, 0.0024215682242336375, 0.00273292535165579],
        "std": [0.21282890555483014, 0.03603655236239526, 0.044628622076697336, 0.014933934907838183]
    }
    motion_vectors = {
        "mean": [[0.0, 0.3191829975906806, -0.38650262600788565],
                 [0.0, 0.2118274829186987, -0.16068544777132687]],
        "std": [[1.0, 1.279285802893745, 4.7578745524065615],
                [1.0, 0.9425657096098303, 2.9294113819454313]]
    }


# mpeg4 dense, no keyframes, training data, only static cameras
# codec = mpeg4, mvs_mode = dense, static_only = True, exclude_keyframes = True, scales = [1.0]
class StatsMpeg4DenseStaticSinglescale():
    velocities = {
        "mean": [0.0016523574430460121, 8.769333919036941e-05],
        "std": [0.045971264593160685, 0.005933205636581293]
    }
    motion_vectors = {
        "mean": [[0.0, 0.016107642943830696, 0.07541612903921065], []],
        "std": [[1.0, 0.06298750897933865, 0.42173798690940506], []]
    }

# mpeg4 dense, no keyframes, training data, only static cameras
# codec = mpeg4, mvs_mode = dense, static_only = False, exclude_keyframes = True, scales = [1.0]
class StatsMpeg4DenseFullSinglescale():
    velocities = {
        "mean": [-0.008838819279986882, 0.0012495100149675096],
        "std": [0.10043215075885126, 0.017104508898420422]
    }
    motion_vectors = {
        "mean": [[0.0, 0.15313816193039162, -0.17677343284746544], []],
        "std": [[1.0, 0.9635213638095246, 2.935473353982918], []]
    }

# h264 dense, no keyframes, training data, only static cameras
# codec = h264, mvs_mode = dense, static_only = True, exclude_keyframes = True, scales = [1.0]
class StatsH264DenseStaticSinglescale():
    velocities = {
        "mean": [0.0016827660191074628, 9.810878965733302e-05],
        "std": [0.04588518077849422, 0.005918017898705134]
    }
    motion_vectors = {
        "mean": [[0.0, 0.02094569523211732, 0.02612519229745142],
                 [0.0, 0.006995176186852561, 0.044139164662715134]],
        "std": [[1.0, 0.171721860009504, 1.5414872371070174],
                [1.0, 0.09525806397511816, 0.590732890938868]]
    }

# h264 dense, no keyframes, training data, full dataset
# codec = h264, mvs_mode = dense, static_only = False, exclude_keyframes = True, scales = [1.0]
class StatsH264DenseFullSinglescale():
    velocities = {
        "mean": [-0.008812261833116142, 0.0012088178726918546],
        "std": [0.10032851314200499, 0.01698779452875495]
    }
    motion_vectors = {
        "mean": [[0.0, 0.8069601918323378, -0.9902591185411875],
                 [0.0, 0.34537754562094747, -0.3013433315617915]],
        "std": [[1.0, 2.5490709017661737, 9.463472385919241],
                 [1.0, 1.4226829001903518, 4.6230886827672]]
    }
