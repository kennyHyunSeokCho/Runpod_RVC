#!/bin/bash
# 한글 주석: .env 파일 생성 스크립트

echo "Creating .env file..."

cat > .env << 'EOF'
# S3 설정 (실제 값으로 변경하세요)
S3_BUCKET=your-bucket-name
S3_REGION=ap-southeast-2
AWS_ACCESS_KEY_ID=your-access-key-id
AWS_SECRET_ACCESS_KEY=your-secret-access-key
AWS_DEFAULT_REGION=ap-southeast-2

# 데이터/결과 프리픽스 (템플릿에서 {uid} 사용 가능)
S3_DATA_PREFIX=voice_blend/uploads/{uid}.mp3
S3_MODELS_PREFIX=voice_blend/models/{uid}

# 고정 리소스 설정
DEFAULT_GPU=0
DEFAULT_BATCH_SIZE=8
EOF

echo ".env file created successfully!"
echo "Please edit .env file with your actual S3 credentials and bucket name."
