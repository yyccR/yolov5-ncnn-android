// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <android/log.h>
#include <jni.h>

#include "layer.h"
#include "net.h"

#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#endif

#include <float.h>
#include <stdio.h>
#include <vector>

static ncnn::Net yolov5;
static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;


struct Object {
    float x;
    float y;
    float w;
    float h;
    int label;
    float prob;
};

static inline float intersection_area(const Object &a, const Object &b) {
    if (a.x > b.x + b.w || a.x + a.w < b.x || a.y > b.y + b.h || a.y + a.h < b.y) {
        // no intersection
        return 0.f;
    }

    float inter_width = std::min(a.x + a.w, b.x + b.w) - std::max(a.x, b.x);
    float inter_height = std::min(a.y + a.h, b.y + b.h) - std::max(a.y, b.y);

    return inter_width * inter_height;
}

static void qsort_descent_inplace(std::vector<Object> &faceobjects, int left, int right) {
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j) {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j) {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object> &faceobjects) {
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object> &faceobjects, std::vector<int> &picked,
                              float nms_threshold) {
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
//        areas[i] = faceobjects[i].rect.area();
        areas[i] = faceobjects[i].w * faceobjects[i].h;
    }

    for (int i = 0; i < n; i++) {
        const Object &a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int) picked.size(); j++) {
            const Object &b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static inline float sigmoid(float x) {
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

static void generate_proposals(const ncnn::Mat &anchors, int stride, const ncnn::Mat &in_pad,
                               const ncnn::Mat &feat_blob, float prob_threshold,
                               std::vector<Object> &objects) {
    const int num_grid_x = feat_blob.w;
    const int num_grid_y = feat_blob.h;

    const int num_anchors = anchors.w / 2;

    const int num_class = feat_blob.c / num_anchors - 5;

    const int feat_offset = num_class + 5;

    for (int q = 0; q < num_anchors; q++) {
        const float anchor_w = anchors[q * 2];
        const float anchor_h = anchors[q * 2 + 1];

        for (int i = 0; i < num_grid_y; i++) {
            for (int j = 0; j < num_grid_x; j++) {
                // find class index with max class score
                int class_index = 0;
                float class_score = -FLT_MAX;
                for (int k = 0; k < num_class; k++) {
//                    float score = sigmoid(feat_blob.channel(q * feat_offset + 5 + k).row(i)[j]);
                    float score = feat_blob.channel(q * feat_offset + 5 + k).row(i)[j];
//                    __android_log_print(ANDROID_LOG_INFO, "ncnn:", "%s", (std::to_string(score).c_str()));

                    if (score > class_score) {
                        class_index = k;
                        class_score = score;
                    }
                }

                float box_score = sigmoid(feat_blob.channel(q * feat_offset + 4).row(i)[j]);

                float confidence = sigmoid(box_score) * sigmoid(class_score);
//                float confidence = box_score * class_score;
                if (confidence >= prob_threshold) {
//                    __android_log_print(ANDROID_LOG_INFO, "ncnn:", "%s", (std::to_string(confidence).c_str()));
                    // yolov5/models/yolo.py Detect forward
                    // y = x[i].sigmoid()
                    // y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                    // y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                    float dx = sigmoid(feat_blob.channel(q * feat_offset + 0).row(i)[j]);
                    float dy = sigmoid(feat_blob.channel(q * feat_offset + 1).row(i)[j]);
                    float dw = sigmoid(feat_blob.channel(q * feat_offset + 2).row(i)[j]);
                    float dh = sigmoid(feat_blob.channel(q * feat_offset + 3).row(i)[j]);

                    float pb_cx = (dx * 2.f - 0.5f + j) * stride;
                    float pb_cy = (dy * 2.f - 0.5f + i) * stride;

                    float pb_w = pow(dw * 2.f, 2) * anchor_w;
                    float pb_h = pow(dh * 2.f, 2) * anchor_h;

                    float x0 = pb_cx - pb_w * 0.5f;
                    float y0 = pb_cy - pb_h * 0.5f;
                    float x1 = pb_cx + pb_w * 0.5f;
                    float y1 = pb_cy + pb_h * 0.5f;

                    Object obj;
                    obj.x = x0;
                    obj.y = y0;
                    obj.w = x1 - x0;
                    obj.h = y1 - y0;
                    obj.label = class_index;
                    obj.prob = confidence;
//                    __android_log_print(ANDROID_LOG_INFO, "ncnn:", "%s", (std::to_string(obj.x).c_str()));


                    objects.push_back(obj);
                }
            }
        }
    }
}

extern "C" {

// FIXME DeleteGlobalRef is missing for objCls
static jclass objCls = NULL;
static jmethodID constructortorId;
static jfieldID xId;
static jfieldID yId;
static jfieldID wId;
static jfieldID hId;
static jfieldID labelId;
static jfieldID probId;


static inline std::string jString2String(JNIEnv *env , jstring jstring)
{
    const char *js =NULL;
    js = env->GetStringUTFChars(jstring, 0);
    std::string s(js);
    env->ReleaseStringUTFChars(jstring, js);
    return s;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_yolov5ncnnandroid_detector_Yolov5NcnnDetector_init(JNIEnv *env, jobject thiz,
                                                                    jobject asset_manager,
                                                                    jstring model_name) {

    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = 4;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;
    opt.use_packing_layout = true;

    // use vulkan compute
    if (ncnn::get_gpu_count() != 0)
        opt.use_vulkan_compute = true;

    AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
    yolov5.opt = opt;

    // init param
    {
        std::string param = jString2String(env, model_name) + ".param";
        int ret = yolov5.load_param(mgr, param.c_str());
        __android_log_print(ANDROID_LOG_INFO,"ncnn:", "%s", ("load_param: " + param + " success").c_str());
        if (ret != 0) {
            __android_log_print(ANDROID_LOG_DEBUG, "ncnn:", "load_param failed");
            return JNI_FALSE;
        }
    }

    // init bin
    {
        std::string bin = jString2String(env, model_name) + ".bin";
        int ret = yolov5.load_model(mgr, bin.c_str());
        __android_log_print(ANDROID_LOG_INFO, "ncnn:", "%s", ("load_bin: " + bin + " success").c_str());
        if (ret != 0) {
            __android_log_print(ANDROID_LOG_DEBUG, "ncnn:", "load_bin failed");
            return JNI_FALSE;
        }
    }

    // init jni glue
    jclass localObjCls = env->FindClass(
            "com/example/yolov5ncnnandroid/detector/Yolov5NcnnDetector$Obj");
    objCls = reinterpret_cast<jclass>(env->NewGlobalRef(localObjCls));

    constructortorId = env->GetMethodID(objCls, "<init>", "(Lcom/example/yolov5ncnnandroid/detector/Yolov5NcnnDetector;)V");

    xId = env->GetFieldID(objCls, "x", "F");
    yId = env->GetFieldID(objCls, "y", "F");
    wId = env->GetFieldID(objCls, "w", "F");
    hId = env->GetFieldID(objCls, "h", "F");
    labelId = env->GetFieldID(objCls, "label", "I");
    probId = env->GetFieldID(objCls, "prob", "F");

    return JNI_TRUE;
}


extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_example_yolov5ncnnandroid_detector_Yolov5NcnnDetector_detect(JNIEnv *env, jobject thiz,
                                                                      jobject bitmap,
                                                                      jboolean use_gpu) {

    if (use_gpu == JNI_TRUE && ncnn::get_gpu_count() == 0) {
        return NULL;
        //return env->NewStringUTF("no vulkan capable gpu");
    }

    const int target_size = 480;
    const float prob_threshold = 0.5f;
    const float nms_threshold = 0.15f;

    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bitmap, &info);
    const int width = info.width;
    const int height = info.height;
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
        return NULL;

//    int img_w = bgr.cols;
//    int img_h = bgr.rows;

    // yolov5/models/common.py DetectMultiBackend
    const int max_stride = 32;

    // letterbox pad to multiple of max_stride
//    int w = img_w;
//    int h = img_h;
    int w = width;
    int h = height;
    float scale = 1.f;
    if (w > h) {
        scale = (float) target_size / w;
        w = target_size;
        h = h * scale;
    } else {
        scale = (float) target_size / h;
        h = target_size;
        w = w * scale;
    }

//    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);
    ncnn::Mat in = ncnn::Mat::from_android_bitmap_resize(env, bitmap, ncnn::Mat::PIXEL_RGB, w, h);

    // pad to target_size rectangle
    // yolov5/utils/datasets.py letterbox
    int wpad = (w + max_stride - 1) / max_stride * max_stride - w;
    int hpad = (h + max_stride - 1) / max_stride * max_stride - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2,
                           ncnn::BORDER_CONSTANT, 114.f);

    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = yolov5.create_extractor();

    ex.input("in0", in_pad);

    std::vector<Object> proposals;

    // anchor setting from yolov5/models/yolov5s.yaml

    // stride 8
    {
        ncnn::Mat out;
        ex.extract("out0", out);

        ncnn::Mat anchors(6);
        anchors[0] = 10.f;
        anchors[1] = 13.f;
        anchors[2] = 16.f;
        anchors[3] = 30.f;
        anchors[4] = 33.f;
        anchors[5] = 23.f;

        std::vector<Object> objects8;
        generate_proposals(anchors, 8, in_pad, out, prob_threshold, objects8);
//        for(auto & proposal : objects8){
//            __android_log_print(ANDROID_LOG_INFO, "ncnn:", "%s", (std::to_string(proposal.x)+" "+std::to_string(proposal.y)).c_str());
//        }
        proposals.insert(proposals.end(), objects8.begin(), objects8.end());

    }

    // stride 16
    {
        ncnn::Mat out;
        ex.extract("out1", out);

        ncnn::Mat anchors(6);
        anchors[0] = 30.f;
        anchors[1] = 61.f;
        anchors[2] = 62.f;
        anchors[3] = 45.f;
        anchors[4] = 59.f;
        anchors[5] = 119.f;

        std::vector<Object> objects16;
        generate_proposals(anchors, 16, in_pad, out, prob_threshold, objects16);

        proposals.insert(proposals.end(), objects16.begin(), objects16.end());

    }

    // stride 32
    {
        ncnn::Mat out;
        ex.extract("out2", out);

        ncnn::Mat anchors(6);
        anchors[0] = 116.f;
        anchors[1] = 90.f;
        anchors[2] = 156.f;
        anchors[3] = 198.f;
        anchors[4] = 373.f;
        anchors[5] = 326.f;

        std::vector<Object> objects32;
        generate_proposals(anchors, 32, in_pad, out, prob_threshold, objects32);


        proposals.insert(proposals.end(), objects32.begin(), objects32.end());

    }

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);
//    for(auto & proposal : proposals){
//        __android_log_print(ANDROID_LOG_INFO, "ncnn:", "%s",
//                            (std::to_string(proposal.x)+" "+std::to_string(proposal.y)).c_str());
//    }

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    std::vector<Object> objects;
    int count = picked.size();
    objects.resize(count);
    for (int i = 0; i < count; i++) {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].x - (wpad / 2)) / scale;
        float y0 = (objects[i].y - (hpad / 2)) / scale;
        float x1 = (objects[i].x + objects[i].w - (wpad / 2)) / scale;
        float y1 = (objects[i].y + objects[i].h - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float) (width - 1)), 0.f);
        y0 = std::max(std::min(y0, (float) (height - 1)), 0.f);
        x1 = std::max(std::min(x1, (float) (width - 1)), 0.f);
        y1 = std::max(std::min(y1, (float) (height - 1)), 0.f);

        objects[i].x = x0;
        objects[i].y = y0;
        objects[i].w = x1 - x0;
        objects[i].h = y1 - y0;
    }

    jobjectArray jObjArray = env->NewObjectArray(objects.size(), objCls, NULL);
    for (size_t i=0; i<objects.size(); i++)
    {
        jobject jObj = env->NewObject(objCls, constructortorId, thiz);

        env->SetFloatField(jObj, xId, objects[i].x);
        env->SetFloatField(jObj, yId, objects[i].y);
        env->SetFloatField(jObj, wId, objects[i].w);
        env->SetFloatField(jObj, hId, objects[i].h);
        env->SetIntField(jObj, labelId, objects[i].label);
        env->SetFloatField(jObj, probId, objects[i].prob);

        env->SetObjectArrayElement(jObjArray, i, jObj);
    }

    return jObjArray;

}

}

