# Algorithm Framework

## Trick

:fire: **LRU_cache可作用于递归函数，取代memo**

## Data Structure

### Binary Heap

堆排序的基础结构。二叉堆其实就是一种特殊的二叉树（完全二叉树），只不过存储在数组里。一般的链表二叉树，我们操作节点的指针，而在**数组里**，我们把数组索引作为指针

```python
def parent(child: int):
  	return child // 2
def left(root: int):
  	return root*2
def right(root: int):
  	return root*2+1
```

### Stack

栈（stack）是很简单的一种数据结构，先进后出的逻辑顺序，符合某些问题的特点，比如说函数调用栈。

单调栈实际上就是栈，只是利用了一些巧妙的逻辑，使得每次新元素入栈后，栈内的元素都保持有序（单调递增或单调递减）。

#### 单调栈

[496.下一个更大元素I](https://leetcode-cn.com/problems/next-greater-element-i)

[503.下一个更大元素II](https://leetcode-cn.com/problems/next-greater-element-ii)

[1118.一月有多少天](https://leetcode-cn.com/problems/number-of-days-in-a-month)

```python
def nextGreaterElement(nums: List) -> List:
  	n = len(nums)
    stack = []  # 保证栈内元素单调性
    res = [0] * n
    # 倒着入栈
    for i in range(n-1, -1, -1):
      	# 若stack.top小于等于nums[i]，将top pop出
      	while stack and stack[-1] <= nums[i]:
          	stack.pop()
        # 返回栈顶
        res[i] = -1 if not stack else stack[-1]
        # 将当前加入栈中
        stack.append(nums[i])
    return res
```

#### 单调队列

[239.滑动窗口最大值](https://leetcode-cn.com/problems/sliding-window-maximum)

```python
from collections import deque
class MonoticQueue():
  	'''
  	保证队列内元素单调性
  	'''
    def __init__(self):
        self.queue = deque()

    def push(self, n):
      	'''
      	Time: <O(n)
      	'''
        # 将小于n的全部pop出
        while self.queue and self.queue[-1] < n:
            self.queue.pop()
        self.queue.append(n)

    def max(self):
      	'''
      	Time: O(1)
      	'''
        # 返回队首
        return self.queue[0]

    def remove(self, n):
      	'''
      	Time: O(1)
      	'''
        # 如果对首为需要remove元素，则deque.popleft()
        if n == self.max():
            self.queue.popleft()
```

### Linked List

```python
'''
基本的单链表节点
'''
class Node:
		def __init__(self, val):
      self.val = val
      self.next = None
      
    def traverse(self):
      # node.val
      while node:
        node = node.next
        
    def traverse(self):
      # nodel.val
      self.traverse(node.next)
```

### Binary Tree

```python
'''
基本的二叉树节点
'''
class TreeNode:
  def __init__(self, val):
    self.val = val
    self.left = None
    self.right = None

	def traverse(self):
    # 前序遍历
    self.traverse(self.left)
    # 中序遍历
    self.traverse(self.right)
    # 后序遍历
    
'''
N叉树
'''
class NTreeNode:
  def __init__(self, val):
    self.val = val
    self.children = []

	def traverse(self):
    for child in children:
      self.traverse(child)
```

**写树相关的算法，简单说就是，先搞清楚当前`root`节点该做什么，然后根据函数定义递归调用子节点**，递归调用会让孩子节点做相同的事情。

:fire:**递归算法的关键要明确函数的定义，相信这个定义，而不要跳进递归细节。**

[116. 填充二叉树节点的右侧指针](https://leetcode-cn.com/problems/populating-next-right-pointers-in-each-node/)

**二叉树的问题难点在于，如何把题目的要求细化成每个节点需要做的事情**，但是如果只依赖一个节点的话，肯定是没办法连接「跨父节点」的两个相邻节点的。那么，我们的做法就是增加函数参数，一个节点做不到，我们就给他安排两个节点，「将每一层二叉树节点连接起来」可以细化成「将每两个相邻节点都连接起来」：

```python
class Solution:
    def connect_two_node(self, node1, node2):
      	'''
      	增加参数来帮助递归
      	'''
        if not node1 or not node2:
            return
        node1.next = node2
        # 状态1：节点1的左右连接
        self.connect_two_node(node1.left, node1.right)
        # 状态2：节点2的左右连接
        self.connect_two_node(node2.left, node2.right)
        # 状态3：节点1的右与节点2左连接
        self.connect_two_node(node1.right, node2.left)

    def connect(self, root: 'Node') -> 'Node':
        if not root:
            return
        self.connect_two_node(root.left, root.right)
        return root
```

[114. 二叉树展开链表](https://leetcode-cn.com/problems/flatten-binary-tree-to-linked-list/)

```python
def flatten(self, root: TreeNode) -> None:
    """
    Do not return anything, modify root in-place instead.
    """
    if not root:
        return

    self.flatten(root.left)
    self.flatten(root.right)

    l = root.left
    r = root.right

    root.left = None
    root.right = l

    p = root
    while p.right:
        p = p.right
    p.right = r
    return root
```

你看，这就是递归的魅力，你说`flatten`函数是怎么把左右子树拉平的？不容易说清楚，**但是只要知道`flatten`的定义如此，相信这个定义，让`root`做它该做的事情，然后`flatten`函数就会按照定义工作。**

另外注意递归框架是后序遍历，因为我们要先拉平左右子树才能进行后续操作。

[106.从中序遍历序列和后序遍历构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/description/)

递归解决，主要侧重当前节点的操作逻辑，然后递归解决子节点。

```python
def buildTree(self, inorder: List[int], postorder: List[int]) -> TreeNode:
    '''
    
    '''
    if not inorder:
        return None
    root = TreeNode()
    # 确定root节点，和在inorder index
    root.val = postorder[-1]
    idx = inorder.index(root.val)
		# 确保left_chunk和right_chunk的大小
    left_chunk_sz = len(inorder[:idx])
    right_chunk_sz = len(inorder[idx+1:])
		# 确定left，right在各自order下的index，递归解决
    root.left = self.buildTree(inorder[:idx], postorder[:left_chunk_sz])
    root.right = self.buildTree(
        inorder[idx+1:], postorder[left_chunk_sz:left_chunk_sz+right_chunk_sz])
    return root
```

### Binary Search Tree

[450.删除二叉搜索树中的节点](https://leetcode-cn.com/problems/delete-node-in-a-bst)

[701.二叉搜索树中的插入操作](https://leetcode-cn.com/problems/insert-into-a-binary-search-tree)

[700.二叉搜索树中的搜索](https://leetcode-cn.com/problems/search-in-a-binary-search-tree)

[98.验证二叉搜索树](https://leetcode-cn.com/problems/validate-binary-search-tree)

**二叉搜索树框架**

```java
void BST(TreeNode root, int target) {
    if (root.val == target)
        // 找到目标，做点什么
    if (root.val < target) 
        BST(root.right, target);
    if (root.val > target)
        BST(root.left, target);
}
```

## Dynamic Programming

**首先，动态规划问题的一般形式就是求最值**。动态规划其实是运筹学的一种最优化方法，只不过在计算机问题上应用比较多，比如说让你求**最长**递增子序列呀，**最小**编辑距离呀等等。

既然是要求最值，核心问题是什么呢？**求解动态规划的核心问题是穷举**。因为要求最值，肯定要把所有可行的答案穷举出来，然后在其中找最值呗。

- 首先，动态规划的穷举有点特别，因为这类问题**存在「重叠子问题」**，如果暴力穷举的话效率会极其低下，所以需要「备忘录」或者「DP table」来优化穷举过程，避免不必要的计算。

- 而且，动态规划问题一定会**具备「最优子结构」**，才能通过子问题的最值得到原问题的最值。

- 另外，虽然动态规划的核心思想就是穷举求最值，但是问题可以千变万化，穷举所有可行解其实并不是一件容易的事，只有列出**正确的「状态转移方程」**才能正确地穷举。

以上提到的**重叠子问题、最优子结构、状态转移方程**就是动态规划三要素。具体什么意思等会会举例详解，但是在实际的算法问题中，**写出状态转移方程是最困难的**，这也就是为什么很多朋友觉得动态规划问题困难的原因，我来提供我研究出来的一个思维框架，辅助你思考状态转移方程：

**明确「状态」-> 明确「选择」 -> 明确 base case -> 定义 dp 数组/函数的含义**。

**方法：**

- **暴力递归**
- **带memory的递归（自顶向下）**
- **dp数组迭代解法（自底向上）**

---

### Subsequence Problem

[53.最大子序和](https://leetcode-cn.com/problems/maximum-subarray/)

[72.编辑距离](https://leetcode-cn.com/problems/edit-distance)

[300.最长上升子序列](https://leetcode-cn.com/problems/longest-increasing-subsequence)

[354.俄罗斯套娃信封问题](https://leetcode-cn.com/problems/russian-doll-envelopes)

[1143.最长公共子序列](https://leetcode-cn.com/problems/longest-common-subsequence)

子序列问题本身就相对子串、子数组更困难一些，**因为前者是不连续的序列，而后两者是连续的**，就算穷举都不容易，更别说求解相关的算法问题了。

1. **第一种思路模板是一个一维的 dp 数组**：

```pseudocode
int n = array.length;
int[] dp = new int[n];

for (int i = 1; i < n; i++) {
    for (int j = 0; j < i; j++) {
        dp[i] = 最值(dp[i], dp[j] + ...)
    }
}
```

举个我们写过的例子 [最长递增子序列](http://mp.weixin.qq.com/s?__biz=MzAxODQxMDM0Mw==&mid=2247484498&idx=1&sn=df58ef249c457dd50ea632f7c2e6e761&chksm=9bd7fa5aaca0734c29bcf7979146359f63f521e3060c2acbf57a4992c887aeebe2a9e4bd8a89&scene=21#wechat_redirect)，在这个思路中 dp 数组的定义是：

**在子数组`array[0..i]`中，以`array[i]`结尾的目标子序列（最长递增子序列）的长度是`dp[i]`**。

2. **第二种思路模板是一个二维的 dp 数组**：

	**2.1** **涉及两个字符串/数组时**（比如最长公共子序列），dp 数组的含义如下：

	**在子数组`arr1[0..i]`和子数组`arr2[0..j]`中，我们要求的子序列（最长公共子序列）长度为`dp[i][j]`**。

	**2.2** **只涉及一个字符串/数组时**（比如本文要讲的最长回文子序列），dp 数组的含义如下：

	**在子数组`array[i..j]`中，我们要求的子序列（最长回文子序列）的长度为`dp[i][j]`**。

```pseudocode
int n = arr.length;
int[][] dp = new dp[n][n];

for (int i = 0; i < n; i++) {
    for (int j = 1; j < n; j++) {
    		# 相等操作
        if (arr[i] == arr[j]) 
            dp[i][j] = dp[i][j] + ...
        # 不相等操作
        else
            dp[i][j] = 最值(...)
    }
}
```

这种思路运用相对更多一些，尤其是涉及两个字符串/数组的子序列。本思路中 dp 数组含义又分为「只涉及一个字符串」和「涉及两个字符串」两种情况。

---

### Greedy

**贪心选择性质呢，简单说就是：每一步都做出一个局部最优的选择，最终的结果就是全局最优。**然而，大部分问题明显不具有贪心选择性质。比如打斗地主，对手出对儿三，按照贪心策略，你应该出尽可能小的牌刚好压制住对方，但现实情况我们甚至可能会出王炸。这种情况就不能用贪心算法，而得使用动态规划解决

#### 重叠区间问题

[435. 无重叠区间](https://leetcode-cn.com/problems/non-overlapping-intervals/)

[452.用最少数量的箭引爆气球](https://leetcode-cn.com/problems/minimum-number-of-arrows-to-burst-balloons)

```python
# 根据尾部进行正序排序
intervals.sort(key=lambda x:x[1])
end = intervals[0][1]
for i in range(1, len(intervals)):
  if end > intervals[i][0]:
    操作
  反向操作
```

#### Jump Game

[55.跳跃游戏](https://leetcode-cn.com/problems/jump-game)

[45.跳跃游戏 II](https://leetcode-cn.com/problems/jump-game-ii)

---

### String Matching Problem

[10.正则表达式匹配](https://leetcode-cn.com/problems/regular-expression-matching/)

如果是两个普通的字符串进行比较，如何进行匹配？我想这个算法应该谁都会写：

**一、普通字符串匹配**

```python
def isMatch(text: str, pattern: str) -> bool:
  # text, pattern索引位置
  i, j = 0, 0
  while j < len(pattern):
    if i >= len(text):
      return False
    if text[i] != pattern[j]:
      return False
    i += 1
    j += 1
  return j == len(i)

def isMatch(text: str, pattern: str) -> bool:
  '''
  递归
  '''
  if not pattern:
    return not text
  firstmatch = bool(text) and text[0] == pattern[0]
  return firstmatch and isMatch(text[1:], pattern[1:])
```

**二、处理点号「.」通配符**

`.` 通配符可以匹配任意**一个**字符

```python
def isMatch(text: str, pattern: str) -> bool:
  '''
  递归
  '''
  if not pattern:
    return not text
  firstmatch = bool(text) and pattern[0] in (text[0], '.')
  return firstmatch and isMatch(text[1:], pattern[1:])
```

**三、处理星号「\*」通配符**

`*` 通配符可以**让前一个字符**出现**任意次数，包括零次**。

```python
def isMatch(text: str, pattern: str) -> bool:
  '''
  递归
  '''
  if not pattern:
    return not text
  firstmatch = bool(text) and pattern[0] in (text[0], '.')
  if len(pattern) >= 2 and pattern[1] == '*':
  	# 发现通配符'*', text与pattern, 返回pattern * 后与text匹配 ｜ firstmatch 和 pattern与text后一位的匹配
    return isMatch(text, pattern[2:]) or firstmatch and isMatch(text[1:], pattern) 
  else:
  	return firstmatch and isMatch(text[1:], pattern[1:])
```

**四、动态规划**

```python
def isMatch(text: str, pattern: str) -> bool:
  '''
  动态规划 with memo
  '''
  def dp(i, j):
    if (i,j) in memo: return memo[(i, j)]
    if j == len(pattern): return i == len(text)
    firstmatch = bool(i < len(text)) and pattern[j] in (text[i], '.')
    if j <= len(pattern)-2 and pattern[j+1] == '*':
      # 发现通配符'*', text与pattern
      ans = dp(i, j+2) or firstmatch and dp(i+1,j)
    else:
      ans = firstmatch and dp(i+1, j+1)
    memo[(i,j)] = ans
    return ans
 	return dp(0, 0)
```

---

### KMP

[28.实现 strStr()](https://leetcode-cn.com/problems/implement-strstr)

**用 `pat` 表示模式串，长度为 `M`，`txt` 表示文本串，长度为 `N`。KMP 算法是在 `txt` 中查找子串 `pat`，如果存在，返回这个子串的起始索引，否则返回 -1**。

暴力的字符串匹配算法很容易写，看一下它的运行逻辑：

```java
// 暴力匹配（伪码）
int search(String pat, String txt) {
    int M = pat.length;
    int N = txt.length;
    for (int i = 0; i <= N - M; i++) {
        int j;
        for (j = 0; j < M; j++) {
            if (pat[j] != txt[i+j])
                break;
        }
        // pat 全都匹配了
        if (j == M) return i;
    }
    // txt 中不存在 pat 子串
    return -1;
}
```

对于暴力算法，如果出现不匹配字符，同时回退 `txt` 和 `pat` 的指针，嵌套 for 循环，时间复杂度 `O(MN)`，空间复杂度`O(1)`。最主要的问题是，如果字符串中重复的字符比较多，该算法就显得很蠢。

明白了 `dp` 数组只和 `pat` 有关，那么我们这样设计 KMP 算法就会比较漂亮：

```java
public class KMP {
    private int[][] dp;
    private String pat;
  
    public KMP(String pat) {
        this.pat = pat;
        // 通过 pat 构建 dp 数组
        // 需要 O(M) 时间
        this.pat = pat;
        int M = pat.length();
        // dp[状态][字符] = 下个状态
        dp = new int[M][256];
        // base case
        dp[0][pat.charAt(0)] = 1;
        // 影子状态 X 初始为 0
        int X = 0;
        // 构建状态转移图（稍改的更紧凑了）
        for (int j = 1; j < M; j++) {
            for (int c = 0; c < 256; c++)
                dp[j][c] = dp[X][c];
            dp[j][pat.charAt(j)] = j + 1;
            // 更新影子状态
            X = dp[X][pat.charAt(j)];
        }
    }

    public int search(String txt) {
        // 借助 dp 数组去匹配 txt
        // 需要 O(N) 时间
        int M = pat.length();
        int N = txt.length();
        // pat 的初始态为 0
        int j = 0;
        for (int i = 0; i < N; i++) {
            // 当前是状态 j，遇到字符 txt[i]，
            // pat 应该转移到哪个状态？
            j = dp[j][txt.charAt(i)];
            // 如果达到终止态，返回匹配开头的索引
            if (j == M) return i - M + 1;
        }
        // 没到达终止态，匹配失败
        return -1;
    }
}
```

---

###  Knapsack Problem

#### 0-1 Knapsack

所以状态有两个，就是「背包的容量」和「可选择的物品」并且可选择物品的数量是**「严格限制」**。 选择就是「装进背包」或者「不装进背包」。

- `dp[i][w]`的定义如下：对于前`i`个物品，当前背包的容量为`w`，这种情况下可以装的最大价值是`dp[i][w]`。

- base case 就是`dp[0][..] = dp[..][0] = 0`

```pseudocode
int dp[N+1][W+1]
dp[0][..] = 0
dp[..][0] = 0

for i in [1..N]:
    for w in [1..W]:
        dp[i][w] = max(
            把物品 i 装进背包,
            不把物品 i 装进背包
        )
return dp[N][W]
```

#### Unbounded Knapsack

有一个背包，最大容量为`amount`，有一系列物品`coins`，每个物品的重量为`coins[i]`，**「每个物品数量无限」**。请问有多少种方法，能够把背包恰好装满？

这个问题和我们前面讲过的两个背包问题，有一个最大的区别就是，每个物品的数量是无限的，这也就是传说中的「**完全背包问题**」，没啥高大上的，无非就是状态转移方程有一点变化而已。

- `dp[i][j]`的定义如下：若只使用前`i`个物品，当背包容量为`j`时，有`dp[i][j]`种方法可以装满背包。

- base case 为`dp[0][..] = 0， dp[..][0] = 1`。因为如果不使用任何硬币面值，就无法凑出任何金额；如果凑出的目标金额为 0，那么“无为而治”就是唯一的一种凑法。

```pseudocode
int dp[N+1][amount+1]
dp[0][..] = 0
dp[..][0] = 1

for i in [1..N]:
    for j in [1..amount]:
        把物品 i 装进背包,
        不把物品 i 装进背包
return dp[N][amount]
```

---

### Stock Problem

[买卖股票的最佳时机](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/solution/) :ballot_box_with_check:

[买卖股票的最佳时机 II](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/) :ballot_box_with_check:

[买卖股票的最佳时机 III :ballot_box_with_check:](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii/)

[买卖股票的最佳时机 IV :ballot_box_with_check:](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iv/)

[最佳买卖股票时机含冷冻期](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/) :ballot_box_with_check:

[买卖股票的最佳时机含手续费 :ballot_box_with_check:](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/)

```mermaid
graph LR;
	HaveStock --sell--> NoStock
	HaveStock --rest--> HaveStock
	NoStock --rest--> NoStock
	NoStock --buy--> HaveStock
	
	
```



```python
'''
dp[i][k][0 or 1]
0 <= i <= n-1, 1 <= k <= K
n 为天数，大 K 为最多交易数
此问题共 n × K × 2 种状态，全部穷举就能搞定。
'''
# base case
for k in range(k+1):
  dp[0][k][0] = 0
  dp[0][k][1] = -prices[0]
# iteration
for 0 <= i < n:
    for 1 <= k <= K:
        for s in {0, 1}:
            dp[i][k][s] = max(buy, sell, rest)
return max(dp[-1][k][0])
```

---

### EGG Problem

[887.鸡蛋掉落](https://leetcode-cn.com/problems/super-egg-drop/)

#### 二分搜索 + 递归 + 剪枝

我们在第`i`层楼扔了鸡蛋之后，可能出现两种情况：鸡蛋碎了，鸡蛋没碎。**注意，这时候状态转移就来了**：

- **如果鸡蛋碎了**，那么鸡蛋的个数`K`应该减一，搜索的楼层区间应该从`[1..N]`变为`[1..i-1]`共`i-1`层楼；

- **如果鸡蛋没碎**，那么鸡蛋的个数`K`不变，搜索的楼层区间应该从 `[1..N]`变为`[i+1..N]`共`N-i`层楼。

因为我们要求的是**最坏情况**下扔鸡蛋的次数，所以鸡蛋在第`i`层楼碎没碎，取决于那种情况的结果**更大**

递归的 **base case** 很容易理解：

- 当楼层数`N`等于 0 时，显然不需要扔鸡蛋；
- 当鸡蛋数`K`为 1 时，显然只能线性扫描所有楼层

```python
'''
[状态]：鸡蛋个数K，需要测试楼层N
[选择]：去哪层楼扔鸡蛋
返回当前状态最优结果
'''
def superEggDrop(self, K: int, N: int) -> int:
    memo = dict()
    def dp(K, N):
        if K == 1:
            return N
        if N == 0:
            return 0
        if (K, N) in memo:
            return memo[(K, N)]
        res = float('INF')
        lo, hi = 1, N
        # 二分搜索优化
        while lo <= hi:
            mid = lo + (hi - lo)//2
            broke = dp(K-1, mid-1)
            not_broke = dp(K, N-mid)
            # 如果鸡蛋碎了，往下继续搜
            if broke > not_broke:
                hi = mid-1
                res = min(res, broke+1)
            # 反之，往上搜
            else:
                lo = mid+1
                res = min(res, not_broke+1)
        memo[(K, N)] = res
        return res
    return dp(K, N)
```

#### DP

现在，我们稍微修改`dp`数组的定义，**确定当前的鸡蛋个数和最多允许的扔鸡蛋次数，就知道能够确定`F`的最高楼层数**。

```python
dp[k][m] = n
# 当前有 k 个鸡蛋，可以尝试扔 m 次鸡蛋
# 这个状态下，最坏情况下最多能确切测试一栋 n 层的楼

# 比如说 dp[1][7] = 7 表示：
# 现在有 1 个鸡蛋，允许你扔 7 次;
# 这个状态下最多给你 7 层楼，
# 使得你可以确定楼层 F 使得鸡蛋恰好摔不碎
# （一层一层线性探查嘛）
```

**1、无论你在哪层楼扔鸡蛋，鸡蛋只可能摔碎或者没摔碎，碎了的话就测楼下，没碎的话就测楼上**。

**2、无论你上楼还是下楼，总的楼层数 = 楼上的楼层数 + 楼下的楼层数 + 1（当前这层楼）**。

根据这个特点，可以写出下面的状态转移方程：

```python
dp[k][m] = dp[k][m-1] + dp[k-1][m-1] + 1
```

**`dp[k][m - 1]`就是楼上的楼层数**，因为鸡蛋个数`k`不变，也就是鸡蛋没碎，扔鸡蛋次数`m`减一；

**`dp[k - 1][m - 1]`就是楼下的楼层数**，因为鸡蛋个数`k`减一，也就是鸡蛋碎了，同时扔鸡蛋次数`m`减一。

PS：这个`m`为什么要减一而不是加一？之前定义得很清楚，这个`m`是一个允许的次数上界，而不是扔了几次。

```python
'''
动态规划
'''
def superEggDrop(self, K: int, N: int) -> int:
    m = 0
    dp = [[0 for _ in range(N+1)] for _ in range(K+1)]
    while dp[K][m] < N:
        m += 1
        for i in range(1, K+1):
            dp[i][m] = dp[i][m-1] + dp[i-1][m-1] + 1
    return m
```

## Backtrack

**解决一个回溯问题，实际上就是一个决策树的遍历过程**。你只需要思考 3 个问题：

1、路径：也就是已经做出的选择。

2、选择列表：也就是你当前可以做的选择。

3、结束条件：也就是到达决策树底层，无法再做选择的条件。

```python
'''
其核心就是 for 循环里面的递归，在递归调用之前「做选择」，在递归调用之后「撤销选择」，特别简单。
'''
result = []
def backtrack(路径, 选择列表):
    if 满足结束条件:
        result.add(路径)
        return

    for 选择 in 选择列表:
      # 做选择
      将该选择从选择列表移除
      路径.add(选择)
      backtrack(路径, 选择列表)
      # 撤销选择
      路径.remove(选择)
      将该选择再加入选择列表
```

#### Conclusion

回溯算法就是个多叉树的遍历问题，关键就是在前序遍历和后序遍历的位置做一些操作，算法框架如下：

```
def backtrack(...):
    for 选择 in 选择列表:
        做选择
        backtrack(...)
        撤销选择
```

**写** **`backtrack`** **函数时，需要维护走过的「路径」和当前可以做的「选择列表」，当触发「结束条件」时，将「路径」记入结果集**。

其实想想看，回溯算法和动态规划是不是有点像呢？我们在动态规划系列文章中多次强调，动态规划的三个需要明确的点就是「状态」「选择」和「base case」，是不是就对应着走过的「路径」，当前的「选择列表」和「结束条件」？

某种程度上说，动态规划的暴力求解阶段就是回溯算法。只是有的问题具有重叠子问题性质，可以用 dp table 或者备忘录优化，将递归树大幅剪枝，这就变成了动态规划。而今天的两个问题，都没有重叠子问题，也就是回溯算法问题了，复杂度非常高是不可避免的。

## Breath First Search

BFS 相对 DFS 的最主要的区别是：**BFS 找到的路径一定是最短的，但代价就是空间复杂度比 DFS 大很多**，至于为什么，我们后面介绍了框架就很容易看出来了。

**问题的本质就是让你在一幅「图」中找到从起点** **`start`** **到终点** **`target`** **的最近距离，这个例子听起来很枯燥，但是 BFS 算法问题其实都是在干这个事儿**

```python
'''
计算从起点 start 到终点 target 的最近距离
'''
def BFS(start, target):
    queue = []
    visited = set()
    queue.append(start)
    visited.add(start)
    step = 0  # 记录扩散的步数
    while queue:
        sz = len(queue)
        # 将当前队列中的所有节点向四周扩散
        for i in range(sz) {
            cur = queue.pop()
            # 划重点：这里判断是否到达终点
            if cur is target:
                return step
            # 将 cur 的相邻节点加入队列, adj()为cur上下左右的点
            for x in cur.adj():
                if x not in visited:
                    queue.append(x)
                    visited.add(x)
        # 划重点：更新步数在这里
        step += 1
```

## Binary Search

```python
def binarySearch(nums: list, target: int):
    left, right = 0, len(nums)-1
    while ___ :
        mid = left + (right - left) // 2
        if nums[mid] == target:
            ___
        elif nums[mid] < target:
            left = ___
        elif nums[mid] > target:
            right = ___
    return ___
```

**分析二分查找的一个技巧是：不要出现 else，而是把所有情况用 else if 写清楚，这样可以清楚地展现所有细节**。本文都会使用 else if，旨在讲清楚，读者理解后可自行简化。

其中 `___` 标记的部分，就是可能出现细节问题的地方，当你见到一个二分查找的代码时，首先注意这几个地方。后文用实例分析这些地方能有什么样的变化。

另外声明一下，计算 mid 时需要防止溢出，代码中 `left + (right - left) / 2` 就和 `(left + right) / 2` 的结果相同，但是有效防止了 `left` 和 `right` 太大直接相加导致溢出。

**为什么 while 循环的条件中是 <=，而不是 <**？

答：因为初始化 `right` 的赋值是 `nums.length - 1`，即最后一个元素的索引，而不是 `nums.length`。

这二者可能出现在不同功能的二分查找中，区别是：前者相当于两端都闭区间 `[left, right]`，后者相当于左闭右开区间 `[left, right)`，因为索引大小为 `nums.length` 是越界的。

我们这个算法中使用的是前者 `[left, right]` 两端都闭的区间。**这个区间其实就是每次进行搜索的区间**。

## Sliding Windows

[76.最小覆盖子串](https://leetcode-cn.com/problems/minimum-window-substring) :ballot_box_with_check:

[567.字符串的排列](https://leetcode-cn.com/problems/permutation-in-string)​ :ballot_box_with_check:

[438.找到字符串中所有字母异位词 :ballot_box_with_check:](https://leetcode-cn.com/problems/find-all-anagrams-in-a-string)

[3.无重复字符的最长子串​ :ballot_box_with_check:](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters)

```python
'''
滑动窗口算法框架
'''
def slidingWindow(s: str, t: str) {
		need, window = defaultdict(), defaultdict()
    for c in t: need[c]+=1;
    left, right = 0, 0 
    valid = 0
    while right < len(s):
        # c 是将移入窗口的字符
        char c = s[right];
        # 右移窗口
        right++;
        # 进行窗口内数据的一系列更新
        ...

        '''
        debug 输出的位置
        ''' 
        print("window: [%d, %d)\n", left, right);

        # 判断左侧窗口是否要收缩
        while window needs shrink:
            # d 是将移出窗口的字符
            char d = s[left]
            # 左移窗口
            left += 1
            # 进行窗口内数据的一系列更新
            ...
```

**其中两处** **`...`** **表示的更新窗口数据的地方，到时候你直接往里面填就行了**。

而且，这两个 `...` 处的操作分别是右移和左移窗口更新操作，等会你会发现它们操作是完全对称的。这个算法技巧的时间复杂度是 O(N)，比字符串暴力算法要高效得多。

**:fire:使用滑动窗口的关键是需要知道何时移动右指针扩大窗口，且何时移动左指针缩小窗口。**

## Intervals Problem

[1288.删除被覆盖区间](https://leetcode-cn.com/problems/remove-covered-intervals) :ballot_box_with_check:

[56.区间合并](https://leetcode-cn.com/problems/merge-intervals) :ballot_box_with_check:

[57.插入区间](https://leetcode-cn.com/problems/insert-interval/):ballot_box_with_check:

[986.区间列表的交集](https://leetcode-cn.com/problems/interval-list-intersections) :ballot_box_with_check:

所谓区间问题，就是线段问题，让你合并所有线段、找出线段的交集等等。主要有两个技巧：

**1、排序**。常见的排序方法就是按照区间起点排序，或者先按照起点升序排序，若起点相同，则按照终点降序排序。当然，如果你非要按照终点排序，无非对称操作，本质都是一样的。

**2、画图**。就是说不要偷懒，勤动手，两个区间的相对位置到底有几种可能，不同的相对位置我们的代码应该怎么去处理。

#### IntervalsCovered & IntervalsMerge

```python
class Solution:
    def removeCoveredIntervals(self, intervals: List[List[int]]) -> int:
        n = len(intervals)
        # 优先对left升序，其次对right降序
        intervals.sort(key=lambda x: (x[0], -x[1]))
        left, right = intervals[0][0], intervals[0][1]
        res = 0
        for inter in intervals[1:]:
            # 找到覆盖区域
            if left <= inter[0] and right >= inter[1]:
                ...
            # 找到相交区域，更新右边界
            if right >= inter[0] and right <= inter[1]:
              	...
                right = inter[1]
            # 区域不相交，更新左右边界
            if right < inter[0]:
              	...
                left = inter[0]
                right = inter[1]
        return n - res
```

可以在 `...` 中进行对应操作，

#### IntervalsIntersection

```python
def intervalIntersection(self, A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
        i,j = 0,0  # 初始化双指针
        res = []
        while i < len(A) and j < len(B):
            a1, a2 = A[i][0], A[i][1]
            b1, b2 = B[j][0], B[j][1]
            # 存在相交区间
            if b2 >= a1 and b1 <= a2:
                # 更新
                res.append([max(a1,b1), min(a2,b2)])
            # 指针前进条件
            if b2 > a2:
                i += 1
            else:
                j += 1
        return res
```

## nSums Problem

[1.两数之和](https://leetcode-cn.com/problems/two-sum) :ballot_box_with_check:

[170.两数之和 III - 数据结构设计](https://leetcode-cn.com/problems/two-sum-iii-data-structure-design) :ballot_box_with_check:

[15.三数之和 :ballot_box_with_check:](https://leetcode-cn.com/problems/3sum/)

[18.四数之和 :ballot_box_with_check:](https://leetcode-cn.com/problems/4sum/)

#### Framework

```python
'''
注意：调用这个函数之前一定要先给 nums 排序
'''
def nSumTarget(nums: List[int], n: int, start: int, target: int)->List[List[int]]:
    sz = len(nums)
		res = []
    # 至少是 2Sum，且数组大小不应该小于 n
    if n < 2 or sz < n: return res
    # 2Sum 是 base case
    if n == 2:
        # 双指针那一套操作
        lo, hi = start, sz - 1;
        while lo < hi:
            sum_ = nums[lo] + nums[hi]
            left, right = nums[lo], nums[hi]
            if sum_ < target:
                while lo < hi and nums[lo] == left: lo+=1
            elif sum_ > target:
                while lo < hi and nums[hi] == right: hi-=1
            else:
                res.append([left, right])
                # 过滤重复值
                while lo < hi and nums[lo] == left: lo+=1
                while lo < hi and nums[hi] == right: hi-=1
    else:
        # n > 2 时，递归计算 (n-1)Sum 的结果
        i = start
        while i < sz:
          	# 递归求nSum
            sub = nSumTarget(nums, n - 1, i + 1, target - nums[i]);
            for arr in sub:
                # (n-1)Sum 加上 nums[i] 就是 nSum
                arr.append(nums[i])
                res.append(arr)
            while i < sz - 1 and nums[i] == nums[i + 1]: i+=1
    return res
```

**关键点在于，不能让第一个数重复，至于后面的两个数，我们复用的 `twoSum` 函数会保证它们不重复**。所以代码中必须用一个 while 循环来保证 `3Sum` 中第一个元素不重复。

## Union Find

### 问题介绍

现在我们的 Union-Find 算法主要需要实现这两个 API：

```python
class UF():
    # 将 p 和 q 连接
    def union(p: int, q: int) -> None
    # 判断 p 和 q 是否连通
    def connected(p: int , q: int ) -> bool
    # 返回图中有多少个连通分量
    def count() -> int
```

这里所说的「连通」是一种等价关系，也就是说具有如下三个性质：

1、自反性：节点`p`和`p`是连通的。

2、对称性：如果节点`p`和`q`连通，那么`q`和`p`也连通。

3、传递性：如果节点`p`和`q`连通，`q`和`r`连通，那么`p`和`r`也连通。

判断这种「等价关系」非常实用，比如说编译器判断同一个变量的不同引用，比如社交网络中的朋友圈计算等等。

### 主要思路

我们设定树的每个节点有一个指针指向其父节点，如果是根节点的话，这个指针指向自己。比如说刚才那幅 10 个节点的图，一开始的时候没有相互连通，就是这样：

![img](https://gblobscdn.gitbook.com/assets%2F-LrtQOWSnDdXhp3kYN4k%2Fsync%2Fd3c6051348983ce3b226f6951a5ceaff17991021.jpg?alt=media)



```python
class UF():
  	def __init__(self, n:int):
      	# 一开始互不连通
      	self.count = n
        self.parent = [0] * n
        #	父节点指针初始指向自己
        for i in range(n):
          	self.parent[i] = i
    # 将 p 和 q 连接
    def union(p: int, q: int) -> None
    # 判断 p 和 q 是否连通
    def connected(p: int , q: int ) -> bool
    # 返回图中有多少个连通分量
    def count() -> int
```

**如果某两个节点被连通，则让其中的（任意）一个节点的根节点接到另一个节点的根节点上**：

![img](https://gblobscdn.gitbook.com/assets%2F-LrtQOWSnDdXhp3kYN4k%2Fsync%2F400479177833924a24f303131950b2b7775ab597.jpg?alt=media)



```python
def union(p: int, q: int)-> None:
    rootP = find(p)
    rootQ = find(q)
    if rootP == rootQ
        return
    # 将两棵树合并为一棵
    self.parent[rootP] = rootQ  # parent[rootQ] = rootP 也一样
    self.count -= 1  # 两个分量合二为一

# 返回某个节点 x 的根节点
def find(x: int)->int:
    # 根节点的 parent[x] == x
    while self.parent[x] != x
    		# 路径压缩！！！O(n)->O(1)
    		self.parent[x] = self.parent[self.parent[x]]
        x = self.parent[x]
    return x
```

**这样，如果节点`p`和`q`连通的话，它们一定拥有相同的根节点**：

![img](https://gblobscdn.gitbook.com/assets%2F-LrtQOWSnDdXhp3kYN4k%2Fsync%2F273302945cdd2c7b45eb4f35a03714368eef5be9.jpg?alt=media)

```python
def connected(p: int, q: int)->bool: 
    rootP = find(p)
    rootQ = find(q)
    return rootP == rootQ
```

### 应用场景

1. 替代DFS

[130.被围绕的区域](https://leetcode-cn.com/problems/surrounded-regions)

**你可以把那些不需要被替换的** **`O`** **看成一个拥有独门绝技的门派，它们有一个共同祖师爷叫** **`dummy`，这些 **`O`**和** **`dummy`** **互相连通，而那些需要被替换的** **`O`** **与** **`dummy`** **不连通**。首先要解决的是，根据我们的实现，Union-Find 底层用的是**一维数组**，构造函数需要传入这个数组的大小，而题目给的是一个二维棋盘。

[990.等式方程的可满足性](https://leetcode-cn.com/problems/surrounded-regions)

使用 Union-Find 算法，主要是如何把原问题转化成图的动态连通性问题。对于算式合法性问题，可以直接利用等价关系，对于棋盘包围问题，则是利用一个虚拟节点，营造出动态连通特性。

另外，将二维数组映射到一维数组，利用方向数组 `d` 来简化代码量，都是在写算法时常用的一些小技巧，如果没见过可以注意一下。

