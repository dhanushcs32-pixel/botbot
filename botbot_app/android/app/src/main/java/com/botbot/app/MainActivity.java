package com.botbot.app;

import android.content.Intent;
import android.os.Bundle;
import androidx.core.content.ContextCompat;
import com.getcapacitor.BridgeActivity;

public class MainActivity extends BridgeActivity {
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        
        // Start the background service with foreground compatibility
        Intent intent = new Intent(this, BotForegroundService.class);
        ContextCompat.startForegroundService(this, intent);
    }
}

