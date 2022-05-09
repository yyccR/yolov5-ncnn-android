package com.example.yolov5ncnnandroid.detector;

import android.app.Activity;
import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;


public class Yolov5NcnnDetector {

    public class Obj {
        public float x;
        public float y;
        public float w;
        public float h;
        public int label;
        public float prob;
    }

    private final String[] LABELS = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
    };
    private final String MODEL_YOLOV5S = "yolov5s.torchscript.ncnn";
    private final String MODEL_YOLOV5N = "yolov5s.torchscript.ncnn";
    private final String MODEL_YOLOV5M = "yolov5s.torchscript.ncnn";
    private String MODEL_FILE;

    public Yolov5NcnnDetector(Activity activity, String modelName) {
        setModelFile(modelName);
        init(activity.getAssets(), getModelFile());
        Log.i("ncnn:","initial model success!");
    }

    public String getModelFile() {
        return this.MODEL_FILE;
    }

    public String getLabel(int labelId){
        if(labelId < LABELS.length){
            return LABELS[labelId];
        } else {
            Log.w("ncnn:", "label index: " + labelId + " out of range.");
            return "";
        }
    }

    public void setModelFile(String modelFile){
        switch (modelFile) {
            case "yolov5s":
                MODEL_FILE = MODEL_YOLOV5S;
                break;
            case "yolov5n":
                MODEL_FILE = MODEL_YOLOV5N;
                break;
            case "yolov5m":
                MODEL_FILE = MODEL_YOLOV5M;
                break;
            default:
                Log.i("ncnn:", "Only yolov5s/n/m can be load!");
        }
    }

    public native boolean init(AssetManager assetManager, String modelName);

    public native Obj[] detect(Bitmap bitmap, boolean use_gpu);

    static {
        System.loadLibrary("yolov5ncnnandroid");
    }
}
