package com.example.rfc_v2;

import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.content.PackageManagerCompat;
import androidx.lifecycle.LifecycleOwner;


import com.google.common.util.concurrent.ListenableFuture;

import org.dmg.pmml.PMML;
import org.jpmml.evaluator.ModelEvaluator;
import org.jpmml.evaluator.ModelEvaluatorFactory;

import java.io.InputStream;
import java.util.concurrent.ExecutionException;

import javax.xml.bind.JAXBException;

public class MainActivity extends AppCompatActivity {

    // TAGS
    private static final String TAG = "TORCH_TORCH";

    // For Model
    private ModelEvaluator<?> evaluator;

    // For Camera Setup
    private static final int REQUEST_CODE_PERMISSION = 101;
    private static final String[] REQUIRED_PERMISSION = new String[]{"android.permission.CAMERA"};
    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    private PreviewView previewView;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        previewView = findViewById(R.id.cameraView);

        if(checkPermissions()){
            startCamera();
        }
        else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSION, REQUEST_CODE_PERMISSION);
        }

    }


    // CAMERA  -------------------------------------------------------------------------------------

    // CAMERA CHECK PERMS
    private boolean checkPermissions() {
        for(String permission : REQUIRED_PERMISSION) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED){
                return false;
            }
        }
        return  true;
    }

    // START CAMERA
    private void startCamera() {
        cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                bindPreview(cameraProvider);
            } catch (ExecutionException | InterruptedException e) {
                Log.e(TAG, "Error starting camera: " + e.getMessage());
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void bindPreview(ProcessCameraProvider cameraProvider){
        Preview preview = new Preview.Builder().build();

        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        try {
            cameraProvider.unbindAll();

            Camera camera = cameraProvider.bindToLifecycle(this, cameraSelector, preview);

            Log.d(TAG, "CAMERA BOUND SUCCESSFULLY");
        }
        catch (Exception e) {
            Log.e(TAG, "USE CASE BINDING FAILED: " + e.getMessage());
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CODE_PERMISSION){
            if (checkPermissions()) {
                startCamera();
            }
            else {
                Log.e(TAG, "PERMISSION NOT GRANTED BY THE USER");
                finish();
            }
        }
    }



    // MODEL ---------------------------------------------------------------------------------------

    // LOAD MODEL
    private void loadModel() throws Exception  {
        AssetManager assetManager = getAssets();
        try (InputStream inputStream = assetManager.open("random_forest_model.pmml")) {
            PMML pmmlModel = org.jpmml.model.PMMLUtil.unmarshal(inputStream);

            ModelEvaluatorFactory modelEvaluatorFactory = ModelEvaluatorFactory.newInstance();
            evaluator = modelEvaluatorFactory.newModelEvaluator(pmmlModel, null);

            evaluator.verify();

        } catch (JAXBException e) {
            Log.e(TAG, "Error parsing PMML file", e);
            throw new Exception("Failed to parse PMML file", e);

        } catch (Exception e) {
            Log.e(TAG, "Error loading PMML model", e);
            throw new Exception("Failed to load PMML model", e);
        }
    }
}