package com.example.yolov5ncnnandroid.detector;

import android.content.res.AssetManager;
import android.graphics.Bitmap;

public class Yolov5NcnnDetector {
    public class Obj {
        public float x;
        public float y;
        public float w;
        public float h;
        public String label;
        public float prob;
    }

    public native Boolean Init(AssetManager mgr);
    public native Obj[] Detect(Bitmap bitmap, Boolean use_gpu);

    static {
        System.loadLibrary("yolov5ncnnandroid");
    }
}
