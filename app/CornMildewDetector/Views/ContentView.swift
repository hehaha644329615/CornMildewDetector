import SwiftUI

struct ContentView: View {
    @State private var showCamera = false
    
    var body: some View {
        NavigationView {
            VStack(spacing: 30) {
                // 标题
                VStack(spacing: 12) {
                    Image(systemName: "leaf.circle.fill")
                        .font(.system(size: 70))
                        .foregroundColor(.green)
                    
                    Text("玉米霉变检测")
                        .font(.largeTitle)
                        .fontWeight(.bold)
                    
                    Text("基于 YOLOv8 + CoreML")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                .padding(.top, 50)
                
                // 功能说明
                VStack(alignment: .leading, spacing: 12) {
                    FeatureRow(icon: "camera.fill", text: "实时视频流检测", color: .blue)
                    FeatureRow(icon: "chart.bar.fill", text: "霉变率自动计算", color: .orange)
                    FeatureRow(icon: "square.stack.3d.up.fill", text: "本地 CoreML 推理", color: .purple)
                    FeatureRow(icon: "bolt.fill", text: "ANE 神经网络加速", color: .yellow)
                }
                .padding(.horizontal, 30)
                
                Spacer()
                
                // 开始检测按钮
                Button(action: {
                    showCamera = true
                }) {
                    HStack {
                        Image(systemName: "camera.fill")
                        Text("开始检测")
                            .fontWeight(.semibold)
                    }
                    .font(.title3)
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 16)
                    .background(Color.green)
                    .cornerRadius(12)
                }
                .padding(.horizontal, 30)
                .padding(.bottom, 50)
            }
            .navigationBarHidden(true)
            .fullScreenCover(isPresented: $showCamera) {
                CameraView()
            }
        }
    }
}

struct FeatureRow: View {
    let icon: String
    let text: String
    let color: Color
    
    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .foregroundColor(color)
                .frame(width: 24)
            Text(text)
                .foregroundColor(.primary)
            Spacer()
        }
    }
}

#Preview {
    ContentView()
}
