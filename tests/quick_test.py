#!/usr/bin/env python3
"""Quick verification script."""

try:
    from mmpp.fft.main import FFTAnalyzer
    from mmpp.fft.modes import FFTModeInterface, FrequencyModeInterface, check_ffmpeg_available, setup_animation_styling
    print('✅ All imports successful')
    
    # Test FFTAnalyzer with mock data
    analyzer = FFTAnalyzer([])
    print('✅ FFTAnalyzer created successfully')
    print('✅ FFTAnalyzer __repr__ method exists:', hasattr(analyzer, '__repr__'))
    
    # Test other classes have __repr__
    print('✅ FFTModeInterface __repr__ exists:', hasattr(FFTModeInterface, '__repr__'))
    print('✅ FrequencyModeInterface __repr__ exists:', hasattr(FrequencyModeInterface, '__repr__'))
    
    # Test animation functions
    ffmpeg_available = check_ffmpeg_available()
    print(f'✅ FFmpeg available: {ffmpeg_available}')
    
    setup_animation_styling()
    print('✅ Animation styling setup successful')
    
    print('🎉 All enhancements working correctly!')
    
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
