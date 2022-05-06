package com.example.yolov5ncnnandroid;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.widget.TextView;

import com.example.yolov5ncnnandroid.databinding.ActivityMainBinding;

public class MainActivity_bak extends AppCompatActivity {

    // Used to load the 'yolov5ncnnandroid' library on application startup.
    static {
        System.loadLibrary("yolov5ncnnandroid");
    }

    private ActivityMainBinding binding;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        // Example of a call to a native method
        TextView tv = binding.sampleText;
        tv.setText(stringFromJNI());
    }

    /**
     * A native method that is implemented by the 'yolov5ncnnandroid' native library,
     * which is packaged with this application.
     */
    public native String stringFromJNI();
}