// 背景图加载检测
const backgroundContainer = document.getElementById('backgroundContainer');
// 提取背景图片URL
let bgImageUrl = window.getComputedStyle(backgroundContainer).backgroundImage;
const urlMatch = bgImageUrl.match(/url\(["']?(.*?)["']?\)/);
bgImageUrl = urlMatch[1];
const bgImage = new Image();
bgImage.src = bgImageUrl;
// 加载完成后显示模型加载
bgImage.onload = function () {
    const modelLoading = document.getElementById('modelLoading');
    if (modelLoading) {
        modelLoading.style.display = 'flex';
    }
}