#include <bits/stdc++.h>
using namespace std;

class StockSpanner {
public:
        vector<int> v;
        StockSpanner() {
        }
        
        stack<pair<int, int>> s;
        int next(int price) {
            int res = 1;
            while (!s.empty() && s.top().first <= price) {
                res += s.top().second;
                s.pop();
            }
            s.push({price, res});
            return res;
        }
    };

/**
 * Your StockSpanner object will be instantiated and called as such:
 * StockSpanner* obj = new StockSpanner();
 * int param_1 = obj->next(price);
 */

    int minCost1(string colors, vector<int>& neededTime) {
        int n = colors.size(), res = 0;
        for (int i = 0; i < n - 1; ++i) {
            if (colors[i] == colors[i + 1]) {
                int sz = 1;
                priority_queue<int, vector<int>, greater<int>> pq;
                pq.push(neededTime[i]);
                while (colors[i] == colors[i + 1]) {
                    sz++;
                    pq.push(neededTime[i+1]);
                    ++i;
                }
                while (sz > 1) {
                    res += pq.top();
                    pq.pop();
                    sz--;
                }
            }
        }
        return res;
    }

    int minCost(string colors, vector<int>& neededTime) {
        int n = colors.size(), res = 0;
        for (int i = 0; i < n - 1; ++i) {
            if (colors[i] == colors[i + 1]) {
                int sum_group = neededTime[i], max_group = neededTime[i];
                while (colors[i] == colors[i + 1]) {
                    max_group = max(max_group, neededTime[i + 1]);
                    sum_group += neededTime[i + 1];
                    ++i;
                }
                res += sum_group - max_group;
            }
        }
        return res;
    }

    int longestSubarray(vector<int>& nums) {
        int zeroCount = 0, longestWindow = 0, start = 0;
        for (int i = 0; i < nums.size(); ++i) {
            zeroCount += (nums[i] == 0);
            while (zeroCount > 1) {
                zeroCount -= (nums[start] == 0);
                start++;
            }
            longestWindow = max(longestWindow, i - start);
        }
        return longestWindow;
    }

 void combination3(vector<vector<int>>& res, vector<int>& sum, int k, int n) {
        if (sum.size() == k && n == 0) {
            res.push_back(sum);
            return;
        }
        for (int i = sum.empty() ? 1 : sum.back() + 1; i <= 9; ++i) {
            if ((n - i) < 0) break;
            sum.push_back(i);
            combination3(res, sum, k, n - i);
            sum.pop_back();
        }
}

    vector<vector<int>> combinationSum3(int k, int n) {
        vector<vector<int>> res;
        vector<int> sum;
        combination3(res, sum, k, n);
        return res;
    }

    int getidx(vector<int>& candidates, int x) {
        for (int i = 0; i < candidates.size(); ++i) {
            if (candidates[i] == x) return i;
        }
        return -1;
    }

    void combination(vector<vector<int>>& res, vector<int>& sum, vector<int>& candidates, int target) {
        if (target == 0) {
            res.push_back(sum);
            return;
        }
        for (int i = sum.empty() ? 0 : getidx(candidates, sum.back()); i < candidates.size(); ++i) {
            if (target - candidates[i] < 0) break;
            sum.push_back(candidates[i]);
            combination(res, sum, candidates, target - candidates[i]);
            sum.pop_back();
        }
    }

    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        vector<vector<int>> res;
        vector<int> sum;
        sort(candidates.begin(), candidates.end());
        combination(res, sum, candidates, target);
        return res;
    }

    vector<int> dailyTemperatures(vector<int>& temperatures) {
        int n = temperatures.size();
        stack<int> st;
        vector<int> res(n, 0);
        for (int i = 0; i < n; ++i) {
            if (st.empty()) st.push(i);
            while (!st.empty() && temperatures[i] > temperatures[st.top()]) {
                res[st.top()] = i - st.top();
                st.pop();
            }
            st.push(i);
        }
        while (!st.empty()) {
            res[st.top()] = 0;
            st.pop();
        }
        return res;
    }

    bool sieve[5000007] = {0};
    void init(int n) {
        for (int i = 2; i <= n; ++i ) {
            sieve[i] = true;
        }
        for (int i = 2; i*i <= n; ++i) {
            if (sieve[i]) {
                for (int j = i*i; j <= n; j += i) {
                    sieve[j] = false;
                }
            }
        }
    }

    int countPrimes(int n) {
        init(n);
        int cnt = 0;
        for (int i = 2; i < n; ++i) {
            if (sieve[i]) cnt++;
        }
        return cnt;
    }

    int findContentChildren(vector<int>& g, vector<int>& s) {
        sort(g.begin(), g.end());
        sort(s.begin(), s.end());
        int i = 0, j = 0, res = 0;
        while (i < g.size() && j < s.size()) {
            if (s[j] >= g[i]) {
                i++, j++, res++;
            }
            else {
                j++;
            }
        }
        return res;
    }

    void combination2(vector<vector<int>>& res, vector<int>& sum, vector<int>& sum_idx, vector<int>& candidates, int target, int start) {
        if (target == 0) {
            res.push_back(sum);
            return;
        }
        for (int i = sum.empty() ? 0 : sum_idx.back() + 1; i < candidates.size(); ++i) {
            if (target - candidates[i] < 0) break;
            if (i > start && candidates[i] == candidates[i - 1]) continue;
            sum_idx.push_back(i);
            sum.push_back(candidates[i]);
            combination2(res, sum, sum_idx, candidates, target - candidates[i], i + 1);
            sum_idx.pop_back();
            sum.pop_back();
        }
    }

    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        vector<vector<int>> res;
        vector<int> sum;
        vector<int> sum_idx;
        sort(candidates.begin(), candidates.end());
        combination2(res, sum, sum_idx, candidates, target, 0);
        return res;
    }

    vector<vector<int>> findMatrix(vector<int>& nums) {
        vector<vector<int>> res;
        int n = nums.size(); 
        int idx = 0;
        vector<bool> flag (n, false);
        while (idx < n) {
            vector<int> sub;
            vector<bool> visited (201, false);
            for (int i = 0; i < n; ++i) {
                if ((!flag[i] && !visited[nums[i]])) {
                    sub.push_back(nums[i]);
                    idx++;
                    flag[i] = true;
                    visited[nums[i]] = true;
                }
            }
            res.push_back(sub);
            
        }
        return res;
    }

    bool fullZero(string s) {
        for (int i = 0; i < s.size(); ++i) {
            if (s[i] == '1') return false;
        }
        return true;
    }

    int numberOfBeams(vector<string>& bank) {
        int n = bank.size();
        int res = 0;
        vector<int> helper;
        for (int i = 0; i < n; ++i) {
            if (fullZero(bank[i])) continue;
            int cnt = 0;
            for (char c : bank[i]) {
                if (c == '1') cnt++;
            }
            helper.push_back(cnt);
        }
        if (helper.size() == 1 || helper.size() == 0) return 0;
        for (int i = 0; i < helper.size() - 1; ++i) {
            res += helper[i] * helper[i + 1];
        }
        return res;
    }

    int minOperations(vector<int>& nums) {
        map<int, int> mp;
        for (int i = 0; i < nums.size(); ++i) {
            mp[nums[i]]++;
        }
        int res = 0;
        for (auto x : mp) {
            if (x.second == 1) return -1;
            res += ceil((double)x.second / 3);
        }
        return res;
    }

    int jump(vector<int>& nums) {
        int n = nums.size();
        if (n <= 1) return 0; // Already at the end

        int maxReach = nums[0]; // Farthest index you can reach from the current position
        int end = nums[0];      // End of the current jump
        int steps = 1;          // Number of jumps made so far

        for (int i = 1; i < n - 1; ++i) {
            maxReach = max(maxReach, i + nums[i]);
            
            if (i == end) { // If you reach the end of the current jump
                end = maxReach; // Update the end of the current jump
                steps++;        // Make a jump
            }
        }

        return steps;
    }

    // int jump(vector<int>& nums) {
    //     int n = nums.size();
    //     vector<int> dp(n, 0);
    //     for (int i = n - 2; i >= 0; --i){
    //         if (nums[i] == 0) {
    //             dp[i] = -1;
    //             continue;
    //         }
    //         for (int j = 1; j <= nums[i]; ++j) {
    //             if (i + j == n - 1) {
    //                 dp[i] = min(dp[i] + 1, dp[i + j] + 1);
    //                 break;
    //             }
    //             else {
    //                 if (dp[i + j] == -1) continue;
    //                 dp[i] = dp[i] == 0 ? dp[i + j] + 1 : min(dp[i], dp[i + j] + 1);
    //             }
    //         }
    //         dp[i] = dp[i] == 0 ? -1: dp[i];
    //     }
    //     return dp[0];
    // }

    bool canJump(vector<int>& nums) {
        int n = nums.size();
        int reach = 0;
        for (int i = 0; i < n; ++i) {
            if (reach < i) return false;
            reach = max(reach, i + nums[i]);
        }
        return true;
    }

    int minPathSum(vector<vector<int>>& grid) {
        int n = grid.size(), m = grid[0].size();
        int dp[n][m];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) dp[i][j] = INT_MAX;
        }
        dp[0][0] = grid[0][0];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (i + 1 < n) dp[i + 1][j] = min(dp[i][j] + grid[i + 1][j], dp[i + 1][j]);
                if (j + 1 < m) dp[i][j + 1] = min(dp[i][j] + grid[i][j + 1], dp[i][j + 1]);
            }
        }
        return dp[n - 1][m - 1];
    }

    bool visited[501][501];
    int x_axis[4] = {-1, 1, 0, 0}, y_axis[4] = {0,0,-1,1};
    void dfs(vector<vector<char>>& grid, int m, int n, int x, int y, pair<int,int> lastCell, bool &flag) {
        visited[x][y] = true;
        for (int k = 0; k < 4; k++) {
            int newx = x + x_axis[k], newy = y + y_axis[k];
            if (newx >= 0 && newy >= 0 && newx < m && newy < n) {
                if (newx != lastCell.first || newy != lastCell.second) {
                    if (grid[newx][newy] == grid[x][y]) {
                        if (visited[newx][newy]) {
                            flag = 1;
                            return;
                        }
                        pair<int,int> lastcell = make_pair(x, y);
                        dfs(grid, m, n, newx, newy, lastcell, flag);
                    } 
                }
            }
        }
    }

    bool containsCycle(vector<vector<char>>& grid) {
        int m = grid.size(), n = grid[0].size();
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) visited[i][j] = false;
        }
        bool flag = false;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (!visited[i][j]) {
                    pair<int, int> lastCell = {i, j};
                    dfs(grid, m, n, i, j, lastCell, flag);
                    if (flag) return true;
                }
            }
        }
        return false;
    }

    int minFallingPathSumHelper(vector<vector<int>>& matrix, int r, int c, vector<vector<int>>& dp){
        if(r == matrix.size()-1 and c < matrix[0].size() and c >= 0) return matrix[r][c]; 
        if(c >= matrix[0].size() or c < 0) return INT_MAX;
        
        if(dp[r][c] != INT_MAX) return dp[r][c];
        return dp[r][c] = matrix[r][c] + min(min(minFallingPathSumHelper(matrix, r+1, c-1, dp), minFallingPathSumHelper(matrix, r+1, c, dp)), minFallingPathSumHelper(matrix, r+1, c+1, dp));
        
    }
    int minFallingPathSum(vector<vector<int>>& matrix) {
        int rows = matrix.size(), cols = matrix[0].size();
        vector<vector<int>> dp(rows+1, vector<int>(cols+1, INT_MAX));
        int ans = INT_MAX;
        for(int c=0; c < cols; c++){
            ans = min(ans, minFallingPathSumHelper(matrix, 0, c, dp));
        }
        return ans;
    }

    struct TreeNode {
        int val;
        TreeNode *left;
        TreeNode *right;
        TreeNode() : val(0), left(nullptr), right(nullptr) {}
        TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
        TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
    };

    int helper(TreeNode* root) {
        if (!root) return INT_MAX;
        if (!root->left && !root->right) return 1;
        return 1 + min(helper(root->left), helper(root->right));
    }

    int minDepth(TreeNode* root) {
        if (!root) return 0;
        return helper(root);
    }

    int sumSubarrayMins(vector<int>& arr) {
        int n = arr.size();
        stack<pair<int, int>> stl, str;
        vector<int> left(n), right(n);

        for (int i = 0; i < n; i++) left[i] = i + 1;

        for (int i = 0; i < n; i++) right[i] = n - i;

        for (int i = 0; i < n; i++) {
            while (!stl.empty() && stl.top().first > arr[i]) stl.pop();
            left[i] = (stl.empty()) ? i + 1 : i - stl.top().second;
            stl.push(make_pair(arr[i], i));

            while (!str.empty() && str.top().first > arr[i]) {
                pair p = str.top(); str.pop();
                right[p.second] = i - p.second;
            }
            str.push(make_pair(arr[i], i));
        }

        int mod = 1e9 + 7;
        long long sum = 0;
        for (int i = 0; i < n; i++) {
            long long subsum = (((long long)arr[i] % mod) * (left[i] % mod) * (right[i] % mod)) % mod;
            sum = ((sum%mod) + subsum) % mod;
        }
        return (int)sum;
    }

    int sumSubarrayMaxs(vector<int>& arr) {
        int n = arr.size();
        stack<pair<int, int>> stl, str;
        vector<int> left(n), right(n);

        for (int i = 0; i < n; i++) left[i] = i + 1;

        for (int i = 0; i < n; i++) right[i] = n - i;

        for (int i = 0; i < n; i++) {
            while (!stl.empty() && stl.top().first < arr[i]) stl.pop();
            left[i] = (stl.empty()) ? i + 1 : i - stl.top().second;
            stl.push(make_pair(arr[i], i));

            while (!str.empty() && str.top().first < arr[i]) {
                pair p = str.top(); str.pop();
                right[p.second] = i - p.second;
            }
            str.push(make_pair(arr[i], i));
        }

        int mod = 1e9 + 7;
        long long sum = 0;
        for (int i = 0; i < n; i++) {
            long long subsum = (((long long)arr[i] % mod) * (left[i] % mod) * (right[i] % mod)) % mod;
            sum = ((sum%mod) + subsum) % mod;
        }
        return (int)sum;
    }

    long long subArrayRanges(vector<int>& nums) {
        return sumSubarrayMaxs(nums) - sumSubarrayMins(nums);
    }

    static bool comp(vector<int>& a, vector<int>& b) {
        return a[1] < b[1];
    }

    int findMinArrowShots(vector<vector<int>>& points) {
        int n = points.size(), res = 0;
        if (n == 1) return 1;
        sort(points.begin(), points.end(), comp);
        for (auto x : points){
            for (int y : x) {
                cout << y << " ";
            }
        }
        cout << endl;
        vector<int> prev = points[0];
        int i;
        for (i = 0; i < n - 1; i++) {
            if (prev[1] >= points[i + 1][0]) {
                while (i < n - 1 && prev[1] >= points[i + 1][0]) {
                    i++;
                }
                prev = i < n - 1 ? points[i + 1] : prev;
                res++;
            } 
            else {
                if (i == n - 2) {
                    res += 2;
                }
                else {
                    prev = points[i + 1];
                    res++;
                }
            }
        }
        return res + (i == (n - 1));
    }

    vector<int> sequentialDigits(int l, int h) {
        queue<int> q;
        for(int i = 1; i <= 9; i++) {
            q.push(i);
        }
        vector<int> ret;
        while(!q.empty()) {
            int f = q.front();
            q.pop();
            if(f <= h && f >= l) {
                ret.push_back(f);
            }
            if(f > h)
                break;
            int num = f % 10;
            if(num < 9) {
                q.push(f * 10 + (num + 1));
            }
        }
        return ret;
    }

    string minWindow(string s, string t) {
        vector<int> map(128,0);
        for(auto c: t) map[c]++;
        int counter=t.size(), begin=0, end=0, d=INT_MAX, head=0;
        while(end<s.size()){
            if(map[s[end++]]-->0) counter--; //in t
            while(counter==0){ //valid
                if(end-begin<d)  d=end-(head=begin);
                if(map[s[begin++]]++==0) counter++;  //make it invalid
            }  
        }
        return d==INT_MAX? "" :s.substr(head, d);
    }

    void dfs(string start, string end, map<string, double>&mp, map<string,vector<string>>& graph, double& val, map<string,int>& visited, bool& found) {
        visited[start] = 1;
        if (found == true) return;
        for (auto child : graph[start]) {
            if (visited[child] != 1) {
                val *= mp[start+"->"+child];
                if (end == child) {
                    found = true;
                    return;
                }
                dfs(child, end, mp, graph, val, visited, found);
                if (found) return;
                else val /= mp[start+"->"+child];
            }
        }
    }

    vector<double> calcEquation(vector<vector<string>>& equations, vector<double>& values, vector<vector<string>>& queries) {
        vector<double> ans;
        map<string,double> mp;
        map<string,vector<string>> graph;
        for (int i = 0; i < equations.size(); i++) {
            string u = equations[i][0], v = equations[i][1];
            mp[u+"->"+v] = values[i];
            mp[v+"->"+u] = 1/values[i];
            graph[u].push_back(v);
            graph[v].push_back(u);
        }
        for (int i = 0; i < queries.size(); i++) {
            string start = queries[i][0], end = queries[i][1];
            if (graph.find(start) == graph.end() || graph.find(end) == graph.end()){
                ans.push_back(-1);
            }
            else {
                double val = 1;
                map<string, int> visited;
                bool found = false;
                if (start == end) found = true;
                else {
                    dfs(start, end, mp, graph, val, visited, found);
                }
                if (found) ans.push_back(val);
                else ans.push_back(-1);
            }
        }
        return ans;
    }

    vector<int> largestDivisibleSubset(vector<int>& nums) {
        int n = nums.size(), max = 0, index = 0;
        vector<int> cnt(n), prev(n, -1);
        sort(nums.begin(), nums.end());
        for (int i = 0; i < n; i++) {
            cnt[i] = 1, prev[i] = -1;
            for (int j = i - 1; j >= 0; j--) {
                if (nums[i] % nums[j] == 0) {
                    if (cnt[i] < 1 + cnt[j]) {
                        cnt[i] = 1 + cnt[j];
                        prev[i] = j;
                    }
                }
            }
            if (cnt[i] > max) {
                max = cnt[i];
                index = i;
            }
        }
        vector<int> res;
        while (index != -1) {
            res.push_back(nums[index]);
            index = prev[index];
        }
        return res;
    }

    int furthestBuilding(vector<int>& A, int bricks, int ladders) {
        priority_queue<int> pq;
        for (int i = 0; i < A.size() - 1; i++) {
            int d = A[i + 1] - A[i];
            if (d > 0)
                pq.push(-d);
            if (pq.size() > ladders) {
                bricks += pq.top();
                pq.pop();
            }
            if (bricks < 0)
                return i;
        }
        return A.size() - 1;
    }

    int findJudge(int n, vector<vector<int>>& trust) {
        map<int, int> mp;
        for (int i = 0; i < trust.size(); ++i) {
            if (!mp.count(trust[i][1])) mp[trust[i][1]];
            else mp[trust[i][1]]++;
        }
        for (int i = 1; i <= n; ++i) {
            if (mp[i] == 0) return i;
        }
        return -1;        
    }

    
    int findCheapestPrice(int n, vector<vector<int>>& flights, int src, int dst, int k) {
        vector<vector<pair<int, int>>> adj(101);
        for (int i = 0; i < flights.size(); ++i) {
            adj[flights[i][0]].push_back(make_pair(flights[i][1], flights[i][2]));
        }
        vector<int> cost(n, INT_MAX);
        cost[src] = 0;
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
        pq.push({0, src});
        while (!pq.empty()) {
            pair<int, int> top = pq.top();;
            pq.pop();
            int u = top.second, cst = top.first;
            if (cst > cost[u]) continue;
            for (auto it : adj[u]) {
                int v = it.first, wgt = it.second;
                if (cost[v] > cost[u] + wgt) {
                    cost[v] = cost[u] + wgt;
                    pq.push({cost[v], v});
                }
            }
        }
        return cost[dst];
    }

    vector<vector<string>> suggestedProducts(vector<string>& products, string searchWord) {
        sort(products.begin(), products.end());
        string prefix = "";
        int start, bsStart = 0, n = products.size();
        vector<vector<string>> res;
        for (char c : searchWord) {
            prefix += c;
            start = lower_bound(products.begin() + bsStart, products.end(), prefix) - products.begin();
            res.push_back({});
            for (int i = start; i < min(start + 3, n) && !products[i].compare(0, prefix.length(), prefix); ++i) {
                res.back().push_back(products[i]);
            }
        }
        return res;
    }

    int gcd(int a, int b) {
        if (a == 0) return b;
        if (b == 0) return a;
        return gcd(b, a%b);
    }

    bool bfs(int n, int u, int v, vector<vector<int>>& adj) {
        vector<bool> visited(100007, false);
        visited[u] = true;
        queue<int> q;
        q.push(u);
        while (!q.empty()) {
            int x = q.front();
            q.pop();
            for (int y : adj[x]) {
                if (!visited[y]) {
                    if (y == v) return true;
                    visited[y] = true;
                    q.push(y);
                }
            }
        }
        return false;
    }

    bool canTraverseAllPairs(vector<int>& nums) {
        int n = nums.size();
        vector<vector<int>> adj(n);
        for (int i = 0; i < nums.size() - 1; ++i) {
            for (int j = i + 1; j < nums.size(); j++) {
                if (gcd(nums[i], nums[j]) > 1) {
                    adj[i].push_back(j);
                    adj[j].push_back(i);
                }
            }
        }
        for (int i = 0; i < nums.size() - 1; ++i) {
            for (int j = i + 1; j < nums.size(); j++) {
                if (!bfs(n, i, j, adj)) return false;
            }
        }
        return true;
    }


//Problem 1609. Even-odd tree
class problem1609 {
    public:
    bool isEvenOddTree(TreeNode* root) {
        queue<TreeNode*> q;
        q.push(root);
        bool even = true;
        int prev;
        while (!q.empty()) {
            int sz = q.size();
            prev = even ? INT_MIN : INT_MAX;
            while (sz) {
                TreeNode* curr = q.front();
                q.pop();
                if (even) {
                    if (curr->val % 2 == 0 || curr->val <= prev) return false;
                }
                else {
                    if (curr->val % 2 == 1 || curr->val >= prev) return false;
                }
                prev = curr->val;
                if (curr->left) q.push(curr->left);
                if (curr->right) q.push(curr->right);
                sz--;
            }
            even = !even;
        }
        return true;
    }
};


//Problem 2864. Maximum Odd Binary Number 
class problem2864 {
    public:
    string maximumOddBinaryNumber(string s) {
        ios::sync_with_stdio(false);
        cin.tie(0);
        cout.tie(0);
        int n = s.size(), cnt = 0;
        for (char c : s) 
            if (c == '1') cnt++;
        string res = "";
        for (int i = 0; i < cnt - 1; ++i) res += '1';
        for (int i = 0; i < n - cnt; ++i) res += '0';
        return res + '1';
    }
};


//Problem 1579.Remove-max-number-of-edges-to-keep-graph-fully-traversable
class problem1579 {
    private:
    class UnionFind {
        vector<int> component;
        int distinctComponents;
        public:
            /*
            *   Initially all 'n' nodes are in different components.
            *   e.g. component[2] = 2 i.e. node 2 belong to component 2.
            */
            UnionFind(int n) {
                distinctComponents = n;
                for (int i=0; i<=n; i++) {
                    component.push_back(i);
                }
            }
            
            /*
            *   Returns true when two nodes 'a' and 'b' are initially in different
            *   components. Otherwise returns false.
            */
            bool unite(int a, int b) {       
                if (findComponent(a) == findComponent(b)) {
                    return false;
                }
                component[findComponent(a)] = b;
                distinctComponents--;
                return true;
            }
            
            /*
            *   Returns what component does the node 'a' belong to.
            */
            int findComponent(int a) {
                if (component[a] != a) {
                    component[a] = findComponent(component[a]);
                }
                return component[a];
            }
            
            /*
            *   Are all nodes united into a single component?
            */
            bool united() {
                return distinctComponents == 1;
            }
    };

    public:
// ----------------- Actual Solution --------------
    int maxNumEdgesToRemove(int n, vector<vector<int>>& edges) {
        // Sort edges by their type such that all type 3 edges will be at the beginning.
        sort(edges.begin(), edges.end(), [] (vector<int> &a, vector<int> &b) { return a[0] > b[0]; });
        
        int edgesAdded = 0; // Stores the number of edges added to the initial empty graph.
        
        UnionFind bob(n), alice(n); // Track whether bob and alice can traverse the entire graph,
                                    // are there still more than one distinct components, etc.
        
        for (auto &edge: edges) { // For each edge -
            int type = edge[0], one = edge[1], two = edge[2];
            switch(type) {
                case 3:
                    edgesAdded += (bob.unite(one, two) | alice.unite(one, two));
                    break;
                case 2:
                    edgesAdded += bob.unite(one, two);
                    break;
                case 1:
                    edgesAdded += alice.unite(one, two);
                    break;
            }
        }
        
        return (bob.united() && alice.united()) ? (edges.size()-edgesAdded) : -1; // Yay, solved.
    }
};
    

//  Problem 948.Bag of Tokens
class problem948 {
    public:
    int bagOfTokensScore(vector<int>& tokens, int power) {
        int score = 0;
        sort(tokens.begin(), tokens.end());
        int i = 0, j = tokens.size() - 1;
        while (i <= j) {
            if (power >= tokens[i]) {
                power -= tokens[i];
                score++;
                i++;
            }
            else if (i < j && score){
                power += tokens[j];
                score--;
                j--;
            }
            else break;
        }
        return score;
    }
};


//Problem 1750. Minimum-length-of-string-after-deleting-similar-ends  
class problem1750 {
    public:
    int minimumLength(string s) {
        int res = s.size();
        int i = 0, j = res - 1;
        while (i < j) {
            if (s[i] != s[j]) break;
            while (i < j && s[i] == s[j]) {
                res -= 2;
                ++i, --j;
            }
            while (i > 0 && i <= j && s[i] == s[i - 1]) {
                res--;
                ++i;
            }
            while (j < s.size() - 1 && i < j && s[j] == s[j + 1]) {
                res--;
                --j;
            }
        }
        return res;
    }
};


//problem 930. Binary Subarrays With Sum
class problem930 {
    private:
    //// Helper function to count the number of subarrays with sum at most the given goal
    int slidingWindowsAtMostGoal(vector<int>& nums, int goal) {
        int start = 0, currentSum = 0, totalCnt = 0;
        // Iterate through the array using a sliding window approach
        for (int end = 0; end < nums.size(); end++) {
            currentSum += nums[end];

            // Adjust the window by moving the start pointer to the right
            // until the sum becomes less than or equal to the goal
            while (start <= end && currentSum > goal) {
                currentSum -= nums[start++];
            }

            // Update the total count by adding the length of the current subarray
            totalCnt += end - start + 1;
        }
        return totalCnt;
    }

    public:
    int numSubarraysWithSum(vector<int>& nums, int goal) {
        return slidingWindowsAtMostGoal(nums, goal) - slidingWindowsAtMostGoal(nums, goal - 1);
    }
};


//problem 57. Insert Interval
class problem57 {
public:
    vector<vector<int>> insert(vector<vector<int>>& intervals, vector<int>& newInterval) {
        int n = intervals.size(), i = 0;
        if (n == 0) return vector<vector<int>> {newInterval};
        vector<vector<int>> res;
        while (i < n && newInterval[0] > intervals[i][1]) {
            res.push_back(intervals[i]);
            i++;
        }
        while (i < n && newInterval[1] >= intervals[i][0]) {
            newInterval[0] = min(newInterval[0], intervals[i][0]);
            newInterval[1] = max(newInterval[1], intervals[i][1]);
            i++;
        }
        res.push_back(newInterval);
        while (i < n) {
            res.push_back(intervals[i++]);
        }
        return res;
    }
};


//problem 2958. length-of-longest-subarray-with-at-most-k-frequency
class problem2958 {
public:
    int maxSubarrayLength(vector<int>& nums, int k) {
        map<int, int> mp;
        int left = 0, right = 0, length = 0;
        while (left <= right && right < nums.size()) {
            mp[nums[right]]++;
            while (mp[nums[right]] > k) {
                mp[nums[left]]--;
                left++;
            }
            length = max(length, right - left + 1);
            right++;
        }
        return length;
    }
};


//problem 2962. Count-subarrays-where-max-element-appears-at-least-k-times
class problem2962 {
public:
    long long countSubarrays(vector<int>& nums, int k) {
        map<int, int> mp;
        int maxEle = 0, left = 0, right = 0;
        long long cnt = 0;
        for (int i = 0; i < nums.size(); ++i) {
            maxEle = max(maxEle, nums[i]);
        }
        while (left <= right && right < nums.size()) {
            mp[nums[right]]++;
            while (mp[maxEle] == k) {
                if (nums[left] == maxEle) mp[nums[left]]--;
                left++;
            }
            cnt += left;
            right++;
        }
        return cnt;
    }
};


//problem 992. Subarrays with K Different Integers
class problem992 {
public:
    int slidingWindowAtMost(vector<int>& nums, int k) {
        unordered_map<int, int> mp;
        int left = 0, right = 0, cnt = 0;
        while (left <= right && right < nums.size()) {
            mp[nums[right]]++;
            while (mp.size() > k) {
                mp[nums[left]]--;
                if (mp[nums[left]] == 0) {
                    mp.erase(nums[left]);
                }
                left++;
            }
            cnt += right - left + 1;
            right++;
        }
        return cnt;
    }

    int subarraysWithKDistinct(vector<int>& nums, int k) {
        ios_base::sync_with_stdio(0);
        cin.tie(0);
        cout.tie(0);
        return slidingWindowAtMost(nums, k) - slidingWindowAtMost(nums, k - 1);
    }
};


//problem 2444. Count Subarrays with Fixed Bounds 
class problem2444 {
public:
    long long countSubarrays(vector<int>& A, int minK, int maxK) {
        long res = 0, jbad = -1, jmin = -1, jmax = -1, n = A.size();
        for (int i = 0; i < n; ++i) {
            if (A[i] < minK || A[i] > maxK) jbad = i;
            if (A[i] == minK) jmin = i;
            if (A[i] == maxK) jmax = i;
            res += max(0L, min(jmin, jmax) - jbad);
        }
        return res;
    }
};


//problem 205. Isomorphic Strings
class problem205 {
public:
    bool isIsomorphic(string s, string t) {
        if (s.size() != t.size()) return false;
        unordered_map<char, char> mp;
        for (int i = 0; i < s.size(); ++i) {
            if (!mp[s[i]]) mp[s[i]] = t[i];
            else if (t[i] != mp[s[i]]) return false;
        }
        return true;
    }
};

//problem 1700. Number of Students Unable to Eat Lunch
class problem1700 {
public:
    int countStudents(vector<int>& students, vector<int>& sandwiches) {
        int nb_one_stu = 0, nb_zero_stu = 0, nb_one_sand = 0, nb_zero_sand = 0;
        for (int i = 0; i < students.size(); ++i) {
            if (students[i] == 1) nb_one_stu++;
            else nb_zero_stu++;
        }
        for (int i = 0; i < sandwiches.size(); ++i) {
            if (sandwiches[i] == 1) nb_one_sand++;
            else nb_zero_sand++;
        }
        int i = 0, j = 0;
        while (i < students.size() && j < sandwiches.size()) {
            if (students[i] == sandwiches[j]) {
                if (students[i] == 1) nb_one_stu--, nb_one_sand--;
                else nb_zero_stu--, nb_zero_sand--;
                i++, j++;
            }
            else {
                if ((!nb_one_stu && sandwiches[j] == 1)) {
                    return nb_zero_stu;
                }
                else if (!nb_zero_stu && sandwiches[j] == 0) {
                    return nb_one_stu;
                }
                else {
                    students.push_back(students[i]);
                    i++;
                }
            }
        }
        return 0;
    }
};

//problem 623. Add One Row to Tree
class problem623 {
public:
    TreeNode* addOneRow(TreeNode* root, int val, int depth) {
        if (depth == 1) {
            TreeNode* newroot;
            return newroot = new TreeNode(val, root, nullptr);
        }
        queue<TreeNode*> q;
        q.push(root);
        int cnt = 0;

        while (!q.empty()) {
            int sz = q.size();
            cnt++;
            while (sz--) {
                TreeNode* curr = q.front();
                q.pop();

                if (cnt != depth - 1) {
                    if (curr->left) q.push(curr->left);
                    if (curr->right) q.push(curr->right);
                }
                else {
                    TreeNode* newnodeleft = new TreeNode(val);
                    newnodeleft->left = curr->left;
                    curr->left = newnodeleft;

                    TreeNode* newnoderight = new TreeNode(val);
                    newnoderight->right = curr->right;
                    curr->right = newnoderight;
                }
            }
        }
        return root;
    }
};

//problem 2751. Robot Collisions
class problem2751 {
    public:
    vector<int> survivedRobotsHealths(vector<int>& positions,
                                      vector<int>& healths, string directions) {
        int n = positions.size();
        vector<int> indices(n), result;
        stack<int> stack;

        for (int index = 0; index < n; ++index) {
            indices[index] = index;
        }

        sort(indices.begin(), indices.end(),
             [&](int lhs, int rhs) { return positions[lhs] < positions[rhs]; });

        for (int currentIndex : indices) {
            // Add right-moving robots to the stack
            if (directions[currentIndex] == 'R') {
                stack.push(currentIndex);
            } else {
                while (!stack.empty() && healths[currentIndex] > 0) {
                    // Pop the top robot from the stack for collision check
                    int topIndex = stack.top();
                    stack.pop();

                    // Top robot survives, current robot is destroyed
                    if (healths[topIndex] > healths[currentIndex]) {
                        healths[topIndex] -= 1;
                        healths[currentIndex] = 0;
                        stack.push(topIndex);
                    } else if (healths[topIndex] < healths[currentIndex]) {
                        // Current robot survives, top robot is destroyed
                        healths[currentIndex] -= 1;
                        healths[topIndex] = 0;
                    } else {
                        // Both robots are destroyed
                        healths[currentIndex] = 0;
                        healths[topIndex] = 0;
                    }
                }
            }
        }

        // Collect surviving robots
        for (int index = 0; index < n; ++index) {
            if (healths[index] > 0) {
                result.push_back(healths[index]);
            }
        }
        return result;
    }
};

//problem 3740. Minimum Distance Between Three Equal Elements I
class problem3740 {
    public:
        int minimumDistance(vector<int>& nums) {
            unordered_map<int,int>idx;
            vector<int> next(nums.size(), -1);
            int res = INT_MAX;
            for (int i = nums.size()-1; i>=0; --i) {
                if (idx.count(nums[i])) {
                    next[i] = idx[nums[i]];
                }
                idx[nums[i]] = i;
            }
            for (int i = 0; i < nums.size(); ++i) {
                if (next[i] > 0 && next[next[i]] > 0) res = min(res, next[next[i]]-i);
            }
            return res == INT_MAX ? -1: 2*res;
        }
};

//problem 3742. Maximum Path Score in a Grid
class problem3742 {
public:
    int maxPathScore(vector<vector<int>>& grid, int k) {
        int m = grid.size(), n = grid[0].size();
        vector<vector<vector<int>>> dp(
            m, vector<vector<int>> (
                n, vector<int> (k+1, INT_MIN)
            )
        );
        int cost = grid[0][0] == 0 ? 0: 1;
        if (cost > k) return -1;
        dp[0][0][cost] = grid[0][0];
        for (int j = 1; j < n; ++j) {
            cost = grid[0][j] == 0 ? 0: 1;
            for (int idx = 0; idx <= k; ++idx) {
                if (dp[0][j-1][idx] == INT_MIN) continue;
                if (cost + idx <= k) {
                    dp[0][j][cost + idx] = dp[0][j - 1][idx] + grid[0][j];
                }
            }
        }
        for (int i = 1; i < m; ++i) {
            cost = grid[i][0] == 0 ? 0: 1;
            for (int idx = 0; idx <= k; ++idx) {
                if (dp[i - 1][0][idx] == INT_MIN) continue;
                if (cost + idx <= k) {
                    dp[i][0][cost + idx] = dp[i - 1][0][idx] + grid[i][0]; 
                }
            }
        }
        // debug 
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                for (int idx = 0; idx <= k; ++idx) {
                    cout << "i: " << i << " j: " << j << " idx: " << idx << " -> " << dp[i][j][idx] << endl;
                }
            }
        }
        //
        for (int i = 1; i < m; ++i) {
            for (int j = 1; j < n; ++j) {
                cost = grid[i][j] == 0 ? 0: 1;
                for (int idx = 0; idx <= k; ++idx) {
                    if (dp[i][j - 1][idx] == INT_MIN) continue;
                    if (cost + idx <= k) {
                        dp[i][j][cost + idx] = max(dp[i][j][cost + idx], dp[i][j-1][idx] + grid[i][j]);
                    }
                }
                for (int idx = 0; idx <= k; ++idx) {
                    if (dp[i - 1][j][idx] == INT_MIN) continue;
                    if (cost + idx <= k) {
                        dp[i][j][cost + idx] = max(dp[i][j][cost + idx], dp[i - 1][j][idx] + grid[i][j]);
                    }
                }
            }
        }

        int res = INT_MIN;
        for (int i = 0; i <= k; ++i){
            res = max(res, dp[m - 1][n - 1][i]);
        }
        return res == INT_MIN ? -1 : res;
    }
};

// Problem 1386. Cinema Seat Allocation
class problem1386 {
public:
    int maxNumberOfFamilies(int n, vector<vector<int>>& reservedSeats) {
        int res = 0;
        sort(reservedSeats.begin(), reservedSeats.end());
        int prevRow = reservedSeats[0][0], emptyRows = n;
        for (int i = 0; i < reservedSeats.size(); ++i) {
            bool reservedSet1 = false, reservedSet2 = false, reservedSet3 = false;
            int currRow = reservedSeats[i][0];
            while (currRow <= n && currRow == prevRow) {
                if (reservedSeats[i][1] >= 2 && reservedSeats[i][1] <= 5) reservedSet1 = true;
                if (reservedSeats[i][1] >= 4 && reservedSeats[i][1] <= 7) reservedSet2 = true;
                if (reservedSeats[i][1] >= 6 && reservedSeats[i][1] <= 9) reservedSet3 = true;
                prevRow = currRow;
                currRow = reservedSeats[i][0];
                i++;
            }
            if (!reservedSet1) res++;
            if (!reservedSet2) res++;
            if (!reservedSet3) res++;
            emptyRows--;
        }
        res += emptyRows * 2;
        return res;
    }
};

class problem2265 {
public:
    vector<int> sumAndCountInSubtree(TreeNode* root, int &cnt) {
        if (!root) return {0, 0};
        cnt++;
        if (!root->left && !root->right) return {root->val, 1};
        int sum = root->val;
        if (root->left) {
            sum += sumAndCountInSubtree(root->left, cnt)[0];
        }
        if (root->right) sum += sumAndCountInSubtree(root->right, cnt)[0];
        return {sum, cnt};
    }

    void dfs(TreeNode* root, int &cnt) {
        int cntNodes = 0;
        vector<int> res = sumAndCountInSubtree(root, cntNodes);
        if (res[1] == 0) return;
        int avg = res[0] / res[1];
        if (root->val == avg) cnt++;
        dfs(root->left, cnt);
        cout << "root->left: " << cnt << endl;
        dfs(root->right, cnt);
        cout << "root->right: " << cnt << endl;
    }

    int averageOfSubtree(TreeNode* root) {
        int cnt = 0;
        dfs(root, cnt);
        return cnt;
    }
};

// 1541. 
class problem1541 {
public:
    int minInsertions(string s) {
        int res = 0;
        stack<int> st;
        for (int i = 0; i < s.size(); i++) {
            if (st.empty()) st.push(i);
            else if (s[i] == ')' && s[st.top()] == ')') {
                int curr_top = st.top(); st.pop();
                if (s[st.top()] == '(') st.pop();
                else {
                    st.push(curr_top);
                    st.push(i);
                }
            }  
            else st.push(i);
        }
        cout << "hehe";
        // now the stack includes non-valid parentheses
        // while (!st.empty()) {
        //     if (s[st.top()] == '(') {
        //         res += 2;
        //         st.pop();
        //     }
        //     else {
        //         res += 1;
        //         st.pop();
        //         st.pop();
        //     }
        // }
        return res;
    }
};

// 3619. Count Islands With Value Divisible by K
class problem3619 {
    public:
    int countIslands(vector<vector<int>>& grid, int k) {
        int n = grid.size(), m = grid[0].size();
        int res = 0;
        vector<vector<bool>> visited(n, vector<bool> (m, false));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (!visited[i][j] && grid[i][j] > 0 && bfs(n,m,i,j,grid,visited) % k == 0) {
                    res++;
                }
            }
        }
        return res;
    }

    int bfs(int n, int m, int x, int y, vector<vector<int>>& grid, vector<vector<bool>>& visited) {
        vector<int> x_dir = {-1,1,0,0};
        vector<int> y_dir = {0,0,-1,1};
        queue<pair<int,int>> q;
        visited[x][y] = true;
        q.push({x,y});
        int sum = 0;
        while (!q.empty()) {
            auto& u = q.front();
            q.pop();
            for (int i = 0; i < 4; ++i) {
                int new_x = u.first + x_dir[i], new_y = u.second + y_dir[i];
                if (new_x < 0 || new_y < 0 || new_x >= n || new_y >= m || grid[new_x][new_y] == 0) continue;
                if (!visited[new_x][new_y]) {
                    visited[new_x][new_y] = true;
                    q.push({new_x, new_y});
                    sum += grid[new_x][new_y];
                }
            }
        }
        return sum;
    }
};

// 3882. Minimum XOR Path in a Grid
class problem3882 {
public:
    int minCost(vector<vector<int>>& grid) {
        // classical dp is not suitable for this problem because the bitwise XOR is non-monotonic - an intermediate path with an larger XOR value can later yield a smaller final XOR value when combined with subsequent cell values. Therefore we cannot greedily track only the minimum XOR value at each cell.
        int n = grid.size(), m = grid[0].size();
        vector<vector<vector<bool>>> dp(n, vector<vector<bool>> (m, vector<bool> (1<<10, false)));
        dp[0][0][grid[0][0]] = true;
        for (int i = 1; i < n; ++i) {
            for (int k = 0; k < (1<<10);++k) {
                if (dp[i-1][0][k]) {
                    dp[i][0][k^grid[i][0]] = true;
                }
            }
        }
        for (int i = 1; i < m; ++i) {
            for (int k = 0; k < (1<<10);++k) {
                if (dp[0][i-1][k]) {
                    dp[0][i][k^grid[0][i]] = true;
                }
            }
        }
        for (int i = 1; i < n; ++i) {
            for (int j = 1; j < m; ++j) {
                for (int k = 0; k < (1<<10); ++k) {
                    if (dp[i-1][j][k]) {
                        dp[i][j][k^grid[i][j]] = true;
                    }
                    if (dp[i][j-1][k]) {
                        dp[i][j][k^grid[i][j]] = true;
                    }
                }
            }
        }
        
        for (int i = 0; i < (1<<10); ++i) {
            if (dp[n-1][m-1][i]) {
                return i;
            }
            // cout << i << ": " << dp[n-1][m-1][i] << "\n";
        }
        return 0;
    }
};

// problem 3665. Twisted Mirror Path Count
class problem3665 {
public:
    int uniquePaths(vector<vector<int>>& grid) {
        constexpr int MOD = 1'000'000'007;

        int m = grid.size();
        int n = grid[0].size();

        // f[i][j][0]: enter (i, j) by moving down
        // f[i][j][1]: enter (i, j) by moving right
        vector<vector<array<long long, 2>>> f(
            m, vector<array<long long, 2>>(n, {0, 0}));

        f[0][0][0] = 1;

        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i > 0) {
                    if (grid[i - 1][j] == 0) {
                        f[i][j][0] = (f[i - 1][j][0] + f[i - 1][j][1]) % MOD;
                    } else {
                        f[i][j][0] = f[i - 1][j][1];
                    }
                }

                if (j > 0) {
                    if (grid[i][j - 1] == 0) {
                        f[i][j][1] = (f[i][j - 1][0] + f[i][j - 1][1]) % MOD;
                    } else {
                        f[i][j][1] = f[i][j - 1][0];
                    }
                }
            }
        }

        return (f[m - 1][n - 1][0] + f[m - 1][n - 1][1]) % MOD;
    }
};

// problem 453. Minimum Moves to Equal Array Elements
class problem453 {
public:
    int minMoves(vector<int>& nums) {
        int sum = 0;
        for (int i = 0; i < nums.size(); ++i) sum += nums[i];
        return sum - *min_element(nums.begin(), nums.end()) * nums.size();
    }
};

//problem 2747. Count Zero Request Servers
class problem2747 {
public:
    vector<int> countServers(int n, vector<vector<int>>& logs, int x, vector<int>& queries) {
        sort(logs.begin(), logs.end(), [&](auto a, auto b) {
            return a[1] < b[1];
        });
        vector<int> ids(queries.size());
        for (int i = 0; i < ids.size(); ++i) ids[i] = i;
        sort(ids.begin(), ids.end(), [&](auto a, auto b){
            return queries[a] < queries[b];
        });
        // for (int i = 0; i < logs.size(); ++i) cout << logs[i][0] << " " << logs[i][1] << "\n";
        vector<int> res(queries.size());
        vector<int> cnt(n + 1);
        int left = 0, right = 0, numServerReceive = 0;
        for (int i: ids) {
            while (right < logs.size() && logs[right][1] <= queries[i]) {
                if (cnt[logs[right][0]] == 0) {
                    numServerReceive++;
                }
                cnt[logs[right][0]]++;
                right++;
            }

            while (left < logs.size() && logs[left][1] < queries[i] - x) {
                cnt[logs[left][0]]--;
                if (cnt[logs[left][0]] == 0) {
                    numServerReceive--;
                }    
                left++;
            }
            res[i] = n - numServerReceive;
        }

        return res;
    }
};

//problem 1606. Find Servers That Handled Most Number of Requests
class problem1606 {
public:
    vector<int> busiestServers(int k, vector<int>& arrival, vector<int>& load) {
        // busy: pair<server_idx, time_complete>
        // idle: server_idx
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> busy;
        set<int> idle;
        vector<int> cnt(k,0);
        for (int i = 0; i < k; ++i) idle.insert(i);
        for (int i = 0; i < arrival.size(); ++i) {
            while (!busy.empty() && busy.top().second <= arrival[i]) {
                int server_top = busy.top().first;
                busy.pop();
                idle.insert(server_top);
            }
            if (!idle.empty()) {
                auto server_it = idle.lower_bound(i % k);
                if (server_it == idle.end()) server_it = idle.begin();
                busy.push({*server_it, arrival[i] + load[i]});
                cnt[*server_it]++;
                idle.erase(*server_it);
            }
        }
        int busiest_server = *max_element(cnt.begin(), cnt.end());
        cout << busiest_server << "\n";
        vector<int> res;
        for (int i = 0; i < k; ++i) {
            if (cnt[i] == busiest_server) res.push_back(i);
        }
        return res;
    }
};

// problem 3693. Climbing Stairs II
class problem3693 {
public:
    int climbStairs(int n, vector<int>& costs) {
        vector<int> dp(n+1);
        dp[0] = 0;
        if (n >= 1) dp[1] = dp[0] + costs[0] + 1;
        if (n >= 2) dp[2] = min(dp[0] + costs[1] + 2*2, dp[1] + costs[1] + 1);
        if (n >= 3) dp[3] = min(dp[0] + costs[2] + 3*3, min(dp[1] + costs[2] + 2*2, dp[2] + costs[2] + 1));
        for (int i = 4; i < n + 1; ++i) {
            dp[i] = min(dp[i - 3] + costs[i - 1] + 3*3, min(dp[i - 2] + costs[i - 1] + 2*2, dp[i - 1] + costs[i - 1] + 1));
        } 
        return dp[n];
    }
};

// problem 3789. Minimum Cost to Acquire Required Items
class problem3789 {
public:
    long long minimumCost(int cost1, int cost2, int costBoth, int need1, int need2) {
        // x + z >= need1 --> x >= max(0, need1 - z)
        // y + z >= need2 --> y >= max(0, need2 - z)
        // Cost = x*cost1 + y*cost2 + z*costBoth
        // >= max(0, need1 - z)*cost1 + max(0, need2 - z)*cost2 + z*costBoth
        // = (x + z)cost1 + (y + z)cost2 + z*costResidual
        // >= (max(0, need1 - z) + z)cost1 + (max(0, need2 - z)+z)cost2 + z*costResidual
        // = max(z,need1)*cost1 + max(z,need2)*cost2 + z*costResidual
        // with costResidual = costBoth - cost1 - cost2
        // sol: for z from 0 to max(need1,need2)
        // -> calculate cost = max(z,need1)*cost1 + max(z,need2)*cost2 + z*costResidual -> minCost

        // exp: cost1 = 3, cost2 = 2, costBoth = 1, need1 = 3, need2 = 2
        // z = 0 -> cost = 3*3 + 2*2 + 0*(-4) = 13
        // z = 1 -> cost = 3*3 + 2*2 + 1*(-4) = 9
        // z = 2 -> cost = 3*3 + 2*2 + 2*(-4) = 5
        // z = 3 -> cost = 3*3 + 3*2 - 3*(-4) = 3 <- minCost

        // exp: cost1 = 5, cost2 = 4, costBoth = 15, need1 = 2, need2 = 3
        // z = 0 -> cost = 2*5 + 3*4 + 0*6 = 22
        // z = 1 -> cost = 2*5 + 3*4 + 1*6 = 28
        // z = 2 -> cost = 2*5 + 3*4 + 2*6 = 34
        // z = 3 -> cost = 3*5 + 3*4 + 3*6 = 40

        long long minCost = LLONG_MAX;
        vector<int> point = {0, need1, need2};
        for (int z: point) {
            long long cost = 1ll*max(z,need1)*cost1 + 1ll*max(z,need2)*cost2 + 1ll*z*(costBoth - cost1 - cost2);
            minCost = min(minCost, cost);
        }
        return minCost;
    }
};

// problem 1524. Number of Sub-arrays With Odd Sum
class problem1524 {
public:
    int numOfSubarrays(vector<int>& arr) {
        // Naive approach: O(n^2)
        // optimizer: prefix sum
        // observation: odd sum subarray = even sum - odd sum = odd sum - even sum
        // even sum subarray = even sum - even sum = odd sum - odd sum
        int cnt = 0, n = arr.size(), mod = 1e9+7;
        vector<int> prefixSum(n + 1, 0);
        int cnt_odd_sum = 0, cnt_even_sum = 1; // sum = 0 is even
        for (int i = 1; i < n + 1; ++i) {
            prefixSum[i] = prefixSum[i - 1] + arr[i - 1];
            if (prefixSum[i] & 1) {
                cnt += cnt_even_sum;
                cnt_odd_sum++;
            }
            else {
                cnt += cnt_odd_sum;
                cnt_even_sum++;
            }
            cnt %= mod;
        }
        return cnt;
    }
};

// problem 2914. Minimum number of changes to make binary string beautiful
class problem2914 {
public:
    int minChanges(string s) {
        // 11011001100011101001
        int cnt = 0;
        for (int i = 0; i < s.size(); i+=2) {
            if (s[i] != s[i+1]) cnt++;
        }
        return cnt;
    }
};

void nam_moi_Binh_Ngo() {
    cout << "Chuc mung nam moi 2026 Binh Ngo" << endl;
    cout << "Thanh cong - Hoc gioi - Hoc bong - GPA > 3.6 - Co nguoi yeu - Tang chieu cao:))))" << endl;
}

int main() {
    cout << "I'm Minh. I am studying Bachelor of Computer Science at Ho Chi Minh City University of Technology (HCMUT, VNU-HCM).\n";

    //problem 57 test
    // vector<vector<int>> intervals = {{1,5}};
    // vector<int> newInterval = {6,8};
    // problem57 p57;
    // vector<vector<int>> res = p57.insert(intervals, newInterval);
    // for (auto x : res) {
    //     cout << "[";
    //     for (auto y : x) cout<< y << " ";
    //     cout << "] ";
    // }

    // //problem 2958 test
    // vector<int> nums = {1,2,3,1,3,2,2,1};
    // int k = 2;
    // problem2958 p2958;
    // cout << p2958.maxSubarrayLength(nums, k) << endl;

    //problem 2962 test
    // vector<int> nums = {1,3,2,3,3};
    // int k = 2;
    // problem2962 p2962;
    // cout << p2962.countSubarrays(nums, k) << endl;

    //problem 2444 test
    // vector<int> nums = {1,3,5,2,7,5};
    // int minK = 1, maxK = 5;
    // problem2444 p2444;
    // cout << p2444.countSubarrays(nums, minK, maxK);

    // //problem 205 test
    // string s = "foo", t = "bar";
    // problem205 p205;
    // cout << p205.isIsomorphic(s, t);

    //problem 1700 test
    // vector<int> students = {1,1,1,0,0,1}, sandwiches = {1,0,0,0,1,1};
    // problem1700 p1700;
    // cout << p1700.countStudents(students, sandwiches);

    //problem 2751 test
    // vector<int> positions = {5,4,3,2,1}, healths = {2,17,9,15,10};
    // string directions = "RRRRR";
    // problem2751 p2751;
    // vector<int> res = p2751.survivedRobotsHealths(positions, healths, directions);
    // for (int h : res) {
    //     cout << h << " ";
    // }
    // cout << endl;

    //problem 3740 test
    // vector<int> nums = {1,1,2,3,2,1,2};
    // problem3740 p3740;
    // cout << p3740.minimumDistance(nums) << endl;

    //problem 3742 test
    // vector<vector<int>> grid = {{0, 1}, {2, 0}};
    // problem3742 p3742;
    // cout << p3742.maxPathScore(grid, 1) << endl;

    //problem 1386 test
    // vector<vector<int>> reservedSeats = {{1,2},{1,3},{1,8},{2,6},{3,1},{3,10}};
    // problem1386 p1386;
    // cout << p1386.maxNumberOfFamilies(3, reservedSeats) << endl;

    // nam_moi_Binh_Ngo();

    // problem 2265 test
    // [4,8,5,0,1,null,6]
    // TreeNode* root = new TreeNode(4);
    // root->left = new TreeNode(8);
    // root->right = new TreeNode(5);
    // root->left->left = new TreeNode(0);
    // root->left->right = new TreeNode(1);
    // root->right->right = new TreeNode(6);
    // problem2265 p2265;
    // cout << p2265.averageOfSubtree(root) << endl;
    // return 0;

    // problem 1541 test
    // string s = "(()))";
    // problem1541 p1541;
    // int res = p1541.minInsertions(s);
    // cout << p1541.minInsertions(s) << endl;

    //problem 3619 test
    // vector<vector<int>> grid = {
    // {0,0,0},
    // {0,0,1},
    // {11,0,6},
    // {0,10,2},
    // {0,0,0},
    // {8,0,0}};
    // problem3619 p3619;
    // int k = 19;
    // cout << p3619.countIslands(grid, k) << endl;

    // problem 3882 test
    // vector<vector<int>> grid = {
    //     {26,24,3},
    //     {2,22,17}
    // };
    // problem3882 p3882;
    // cout << p3882.minCost(grid) << endl;

    // problem 1606 test
    int k = 3;
    vector<int> arrival = {1, 2, 3, 4, 5};
    vector<int> load = {5, 2, 3, 3, 3};
    problem1606 p1606;
    vector<int> res = p1606.busiestServers(k, arrival, load);
    for (int server : res) {
        cout << server << " ";
    }
    cout << endl;

    return 0;
}