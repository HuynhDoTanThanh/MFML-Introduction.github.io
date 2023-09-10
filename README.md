# Mathematics for machine learning
# Lesson 1: Introduction
## 1.	What is Machine Learning?
Tom Mitchell described machine learning as: “A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E.”

Machine learning is about designing algorithms that automatically extract valuable information from data. The emphasis here is on “automatic”, i.e., machine learning is concerned with general-purpose methodologies that can be applied to many datasets while producing something that is meaningful. There are three concepts that are at the core of machine learning: data, a model, and learning.

In general, any machine learning problem can be assigned to one of two broad classifications: Supervised learning and Unsupervised learning.
## 2.	Mathematics for Machine Learning
Machine learning builds upon the language of mathematics to express concepts that seem intuitively obvious but that is surprisingly difficult to formalize. Once formalized properly, we can gain insights into the ask we want to solve. This enables us to precisely delineate real-world problems and employ machine learning to address them effectively.
## 3.	Supervised Learning
In supervised learning, we are given a data set and already know what our correct output should look like, having the idea that there is a relationship between the input and the output.

Supervised learning problems are categorized into "regression" and "classification" problems. In a regression problem, we are trying to predict results within a continuous output, meaning that we are trying to map input variables to some continuous function. In a classification problem, we are instead trying to predict results in a discrete output. In other words, we are trying to map input variables into discrete categories. 
### Examples:
Given data about the size of houses on the real estate market, try to predict their price. Price as a function of size is a continuous output, so this is a regression problem. 

We could turn this example into a classification problem by instead making our output about whether the house "sells for more or less than the asking price." Here we are classifying the houses based on price into two discrete categories. 
## 4.	Unsupervised Learning
Unsupervised learning allows us to approach problems with little or no idea what our results should look like. We can derive structure from data where we don't necessarily know the effect of the variables.

We can derive this structure by clustering the data based on relationships among the variables in the data.
With unsupervised learning, there is no feedback based on the prediction results.
### Examples:
Clustering: Take a collection of 1,000,000 different genes and find a way to automatically group these genes into groups that are somehow similar or related by different variables, such as lifespan, location, roles, and so on.

Non-clustering: The "Cocktail Party Algorithm", allows you to find structure in a chaotic environment. (i.e., identifying individual voices and music from a mesh of sounds at a cocktail party).
 
# Bài 1: Introduction
## 1.	Machine Learning là gì?
Có rất nhiều định nghĩa về Machine Learning là gì? Tom Mitchell đã định nghĩa Machine Learning là: “A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E.”

Machine Learning là việc thiết kế một thuật toán trích xuất thông tin có giá trị từ dữ liệu. Điểm cốt lõi ở đây là “tự động”, nghĩa là Machine Learning có thể áp dụng một cách tự động cho nhiều bài toán khác nhau (khác nhau về bộ dữ liệu) nhưng có cùng một mục đích (phân loại, hồi quy) bằng cách cố gắng tạo ra thông tin có ý nghĩa từ bộ dữ liệu. Có ba khái niệm quan trọng trong Machine Learning là: data, model và learning.

Nhìn chung, bất kỳ vấn đề Machine Learning nào cũng có thể được gán cho một trong hai loại: Học có giám sát và Học không giám sát.
## 2.	Toán cho Machine Learning
Machine Learning xây dựng dựa trên ngôn ngữ của toán học để thể hiện các khái niệm có vẻ rõ ràng bằng trực giác nhưng điều đó rất khó để hiểu được bên trong của một mô hình Machine Learning là gì. Sau khi chúng ta biết được cách thức hoạt động của nó, chúng ta có thể hiểu rõ hơn về yêu cầu mà chúng ta muốn giải quyết. Điều này giúp chúng ta xác định được chính xác vấn đề thực tế trong machine learning là gì và có thể áp dụng nó để giải quyết bài toán của mình một cách hiệu quả.
## 3.	Supervised Learning
Trong học có giám sát, chúng ta được cung cấp một tập dữ liệu và đã biết đầu ra chính xác của mình sẽ trông như thế nào, từ đó nảy ra ý tưởng rằng có mối quan hệ giữa đầu vào và đầu ra. 

Các vấn đề học có giám sát được phân thành hai loại "regression" (hồi quy) và "classification" (phân loại). Trong một bài toán hồi quy, chúng ta cố gắng dự đoán kết quả trong một đầu ra liên tục, có nghĩa là chúng ta đang cố gắng ánh xạ các biến đầu vào thành một hàm liên tục nào đó. Còn về bài toán phân loại, chúng ta cố gắng dự đoán kết quả trong một đầu ra rời rạc. Nói cách khác, chúng ta đang cố gắng ánh xạ các biến đầu vào vào các danh mục rời rạc.
### Examples: 
Với dữ liệu về diện tích của các căn nhà trên thị trường bất động sản, cố gắng dự đoán giá của chúng. Giá như một hàm số của diện tích là một đầu ra liên tục, vì vậy đây là một vấn đề hồi quy.

Chúng ta có thể biến ví dụ này thành một vấn đề phân loại bằng cách thay đổi đầu ra của chúng ta để xác định xem căn nhà "bán với giá cao hơn hay thấp hơn giá yêu cầu." Ở đây, chúng ta phân loại các căn nhà dựa trên giá thành hai danh mục rời rạc.


## 4.	Unsupervised Learning
Học không giám sát cho phép chúng ta tiếp cận các vấn đề mà chúng ta có ít hoặc không có ý tưởng về kết quả cuối cùng. Chúng ta có thể tìm ra cấu trúc từ dữ liệu mà chúng ta không nhất thiết phải biết tác động của các biến.

Chúng ta có thể tạo ra cấu trúc này bằng cách gom cụm dữ liệu dựa trên mối quan hệ giữa các biến trong dữ liệu. Trong học không giám sát, không có phản hồi dựa trên kết quả dự đoán.
### Examples:
Gom cụm: Lấy một bộ sưu tập gồm 1,000,000 gen khác nhau và tìm cách tự động gom nhóm các gen này thành các nhóm có sự tương đồng hoặc mối quan hệ nào đó bằng các biến khác nhau, như tuổi thọ, vị trí, vai trò, và vì sao.

Không phải gom cụm: Thuật toán "Cocktail Party", cho phép bạn tìm ra cấu trúc trong môi trường hỗn loạn (ví dụ: xác định giọng nói và âm nhạc riêng biệt trong một mớ âm thanh tại một buổi tiệc cocktail).
