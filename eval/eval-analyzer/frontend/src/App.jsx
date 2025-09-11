import React, { useState, useEffect, useMemo } from 'react';
import { 
  Card, 
  Select, 
  Row, 
  Col, 
  Button, 
  Tag, 
  Statistic, 
  Space, 
  Typography, 
  Divider,
  Alert,
  Image,
  Badge,
  InputNumber,
  Tooltip,
  Switch
} from 'antd';
import { 
  LeftOutlined, 
  RightOutlined, 
  CheckCircleOutlined, 
  CloseCircleOutlined,
  EyeOutlined,
  InfoCircleOutlined
} from '@ant-design/icons';

const { Title, Text, Paragraph } = Typography;
const { Option } = Select;

// 主数据配置
const MAIN_EVAL_FILE_PATH = '/evaluation_results2.json';
const getMainImagePath = (fileName) => `/images2/${fileName}`;

// 对比数据配置
const COMPARE_EVAL_FILE_PATH = '/evaluation_results.json';
const getCompareImagePath = (fileName) => `/images/${fileName}`;

const EvaluationViewer = () => {
  const [mainData, setMainData] = useState([]);
  const [compareData, setCompareData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const [filters, setFilters] = useState({
    ui_type: undefined,
    group: undefined,
    platform: undefined,
    application: undefined,
    hit_success: undefined
  });
  const [currentIndex, setCurrentIndex] = useState(0);
  const [displayOptions, setDisplayOptions] = useState({
    showAttention: false,
    showBbox: true,        // 控制原本的红色边界框
    showHitTop1: false,    // 控制hit_topk_cord第一个坐标点
    showHitTopk: false     // 控制hit_topk_cord所有坐标点
  });

  // 数据加载
  useEffect(() => {
    loadEvaluationData();
  }, []);

  // 匹配对比数据的函数
  const matchCompareData = (mainItem) => {
    if (!mainItem || !compareData.length) return null;
    
    // 从主数据文件名生成对比数据文件名
    // 例如: android_studio_mac/screenshot_2024-11-28_15-16-55.png
    // 转换为: android_studio_mac/screenshot_2024-11-28_15-16-55-beta.png
    const mainFileName = mainItem.file_name;
    const baseFileName = mainFileName.replace(/\.png$/, '');
    const compareFileName = baseFileName + '-beta.png';
    
    // 在对比数据中查找匹配项
    const matchedItem = compareData.find(item => 
      item.file_name === compareFileName ||
      item.img_filename === compareFileName // 兼容不同的字段名
    );
    
    return matchedItem;
  };

  // 获取所有唯一值用于筛选选项（只基于主数据）
  const filterOptions = useMemo(() => {
    return {
      ui_types: [...new Set(mainData.map(item => item.ui_type))].filter(Boolean),
      groups: [...new Set(mainData.map(item => item.group))].filter(Boolean),
      platforms: [...new Set(mainData.map(item => item.platform))].filter(Boolean),
      applications: [...new Set(mainData.map(item => item.application))].filter(Boolean)
    };
  }, [mainData]);

  // 根据筛选条件过滤数据（只筛选主数据）
  const filteredData = useMemo(() => {
    return mainData.filter(item => {
      if (filters.ui_type && item.ui_type !== filters.ui_type) return false;
      if (filters.group && item.group !== filters.group) return false;
      if (filters.platform && item.platform !== filters.platform) return false;
      if (filters.application && item.application !== filters.application) return false;
      if (filters.hit_success === 'success' && item.hit_top1 !== 1) return false;
      if (filters.hit_success === 'failed' && item.hit_top1 !== 0) return false;
      return true;
    });
  }, [mainData, filters]);

  // 统计数据（基于主数据）
  const statistics = useMemo(() => {
    const total = filteredData.length;
    const successful = filteredData.filter(item => item.hit_top1 === 1).length;
    const successRate = total > 0 ? ((successful / total) * 100).toFixed(1) : 0;
    const avgOverlap = total > 0 ? 
      (filteredData.reduce((sum, item) => sum + item.overlap_top1, 0) / total).toFixed(3) : 0;
    
    return { total, successful, successRate, avgOverlap };
  }, [filteredData]);

  // 重置当前索引当筛选条件改变时
  useEffect(() => {
    setCurrentIndex(0);
  }, [filters]);

  const currentMainItem = filteredData[currentIndex];
  const currentCompareItem = currentMainItem ? matchCompareData(currentMainItem) : null;

  const handleFilterChange = (key, value) => {
    setFilters(prev => ({
      ...prev,
      [key]: value
    }));
  };

  const goToNext = () => {
    if (currentIndex < filteredData.length - 1) {
      setCurrentIndex(currentIndex + 1);
    }
  };

  const goToPrevious = () => {
    if (currentIndex > 0) {
      setCurrentIndex(currentIndex - 1);
    }
  };

  const goToPage = (pageIndex) => {
    const index = pageIndex - 1;
    if (index >= 0 && index < filteredData.length) {
      setCurrentIndex(index);
    }
  };

  const getPlatformColor = (platform) => {
    const colors = {
      'macos': 'blue',
      'windows': 'green',
      'linux': 'orange',
      'android': 'purple',
      'ios': 'cyan'
    };
    return colors[platform] || 'default';
  };

  const getUITypeColor = (uiType) => {
    const colors = {
      'icon': 'magenta',
      'button': 'blue',
      'text': 'green',
      'input': 'orange',
      'menu': 'purple'
    };
    return colors[uiType] || 'default';
  };

  const loadEvaluationData = async () => {
    try {
      setLoading(true);
      
      // 并行加载主数据和对比数据
      const [mainResponse, compareResponse] = await Promise.all([
        fetch(MAIN_EVAL_FILE_PATH),
        fetch(COMPARE_EVAL_FILE_PATH)
      ]);
      
      if (!mainResponse.ok) {
        throw new Error('Failed to load main evaluation data');
      }
      if (!compareResponse.ok) {
        throw new Error('Failed to load compare evaluation data');
      }
      
      const [mainJsonData, compareJsonData] = await Promise.all([
        mainResponse.json(),
        compareResponse.json()
      ]);
      
      setMainData(mainJsonData);
      setCompareData(compareJsonData);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const generateAttentionHeatmap = (attnScores, imageWidth, imageHeight, nWidth, nHeight) => {
    if (!attnScores || !Array.isArray(attnScores) || attnScores.length === 0) return null;
    
    try {
      // 将注意力分数重塑为二维数组 (对应Python中的reshape)
      const scores = attnScores[0]; // 取第一个元素，对应Python中的attn_scores[0]
      const scoresArray = [];
      
      for (let i = 0; i < nHeight; i++) {
        const row = [];
        for (let j = 0; j < nWidth; j++) {
          row.push(scores[i * nWidth + j]);
        }
        scoresArray.push(row);
      }
      
      // 找到最小值和最大值进行归一化
      let minScore = Infinity;
      let maxScore = -Infinity;
      for (let i = 0; i < nHeight; i++) {
        for (let j = 0; j < nWidth; j++) {
          minScore = Math.min(minScore, scoresArray[i][j]);
          maxScore = Math.max(maxScore, scoresArray[i][j]);
        }
      }
      
      // 创建归一化的分数映射
      const normalizedScores = [];
      for (let i = 0; i < nHeight; i++) {
        const row = [];
        for (let j = 0; j < nWidth; j++) {
          const normalized = (scoresArray[i][j] - minScore) / (maxScore - minScore);
          row.push(Math.floor(normalized * 255));
        }
        normalizedScores.push(row);
      }
      
      // 创建canvas并绘制热力图
      const canvas = document.createElement('canvas');
      canvas.width = imageWidth;
      canvas.height = imageHeight;
      const ctx = canvas.getContext('2d');
      
      // 创建临时canvas用于缩放
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = nWidth;
      tempCanvas.height = nHeight;
      const tempCtx = tempCanvas.getContext('2d');
      const tempImageData = tempCtx.createImageData(nWidth, nHeight);
      
      // 应用jet colormap（简化版）
      const applyJetColormap = (value) => {
        const normalized = value / 255.0;
        let r, g, b;
        
        if (normalized < 0.25) {
          r = 0;
          g = Math.floor(normalized * 4 * 255);
          b = 255;
        } else if (normalized < 0.5) {
          r = 0;
          g = 255;
          b = Math.floor(255 - (normalized - 0.25) * 4 * 255);
        } else if (normalized < 0.75) {
          r = Math.floor((normalized - 0.5) * 4 * 255);
          g = 255;
          b = 0;
        } else {
          r = 255;
          g = Math.floor(255 - (normalized - 0.75) * 4 * 255);
          b = 0;
        }
        
        return [r, g, b, Math.floor(0.6 * 255)]; // 0.6透明度对应blend alpha=0.3的效果
      };
      
      // 填充临时canvas
      for (let i = 0; i < nHeight; i++) {
        for (let j = 0; j < nWidth; j++) {
          const pixelIndex = (i * nWidth + j) * 4;
          const [r, g, b, a] = applyJetColormap(normalizedScores[i][j]);
          tempImageData.data[pixelIndex] = r;
          tempImageData.data[pixelIndex + 1] = g;
          tempImageData.data[pixelIndex + 2] = b;
          tempImageData.data[pixelIndex + 3] = a;
        }
      }
      
      tempCtx.putImageData(tempImageData, 0, 0);
      
      // 缩放到目标尺寸 (对应Python中的resize，使用NEAREST)
      ctx.imageSmoothingEnabled = false; // 对应NEAREST重采样
      ctx.drawImage(tempCanvas, 0, 0, imageWidth, imageHeight);
      
      return canvas.toDataURL();
    } catch (error) {
      console.error('Error generating attention heatmap:', error);
      return null;
    }
  };

  const ImageWithBbox = ({ item, isCompare = false, title }) => {
    const [imageError, setImageError] = useState(false);

    if (!item) return null;

    const { bbox_x1y1x2y2, img_size, file_name, img_filename, attn_score, hit_topk_cord, n_width_n_height } = item;
    const [x1, y1, x2, y2] = bbox_x1y1x2y2;
    const fileName = file_name || img_filename;
    const imagePath = isCompare ? getCompareImagePath(fileName) : getMainImagePath(fileName);

    // 红色边界框样式
    const bboxStyle = {
      position: 'absolute',
      left: `${x1 * 100}%`,
      top: `${y1 * 100}%`,
      width: `${(x2 - x1) * 100}%`,
      height: `${(y2 - y1) * 100}%`,
      border: '1px solid #ff4d4f',
      backgroundColor: 'rgba(255, 77, 79, 0.1)',
      pointerEvents: 'none',
      borderRadius: '1px'
    };

    // 生成坐标点样式
    const getCoordinatePoints = () => {
      if (!hit_topk_cord || !Array.isArray(hit_topk_cord)) return [];
      
      const colors = ['#52c41a', '#1890ff', '#722ed1', '#eb2f96', '#fa8c16'];
      
      return hit_topk_cord.map((coords, index) => {
        if (!coords || coords.length < 2) return null;
        
        // 假设坐标格式为[x, y]或[x1, y1, x2, y2]，取中心点
        let centerX, centerY;
        if (coords.length === 2) {
          [centerX, centerY] = coords;
        } else if (coords.length === 4) {
          const [x1, y1, x2, y2] = coords;
          centerX = (x1 + x2) / 2;
          centerY = (y1 + y2) / 2;
        } else {
          return null;
        }

        return {
          position: 'absolute',
          left: `${centerX * 100}%`,
          top: `${centerY * 100}%`,
          width: '4px',
          height: '4px',
          borderRadius: '50%',
          backgroundColor: colors[index % colors.length],
          border: '1px solid white',
          transform: 'translate(-50%, -50%)', // 居中对齐
          pointerEvents: 'none',
          zIndex: 10,
          boxShadow: '0 2px 4px rgba(0,0,0,0.3)'
        };
      }).filter(Boolean);
    };

    // 获取第一个坐标点（hit_top1）
    const getTop1Point = () => {
      if (!hit_topk_cord || !Array.isArray(hit_topk_cord) || hit_topk_cord.length === 0) return null;
      
      const coords = hit_topk_cord[0];
      if (!coords || coords.length < 2) return null;
      
      let centerX, centerY;
      if (coords.length === 2) {
        [centerX, centerY] = coords;
      } else if (coords.length === 4) {
        const [x1, y1, x2, y2] = coords;
        centerX = (x1 + x2) / 2;
        centerY = (y1 + y2) / 2;
      } else {
        return null;
      }

      return {
        position: 'absolute',
        left: `${centerX * 100}%`,
        top: `${centerY * 100}%`,
        width: '5px',
        height: '5px',
        borderRadius: '50%',
        backgroundColor: '#52c41a',
        border: '2px solid white',
        transform: 'translate(-50%, -50%)',
        pointerEvents: 'none',
        zIndex: 11,
        boxShadow: '0 3px 6px rgba(0,0,0,0.4)'
      };
    };

    return (
      <div className="relative">
        <div style={{ marginBottom: '12px' }}>
          <Text strong style={{ fontSize: '16px' }}>
            {title}
          </Text>
          <Badge 
            style={{ marginLeft: '8px' }}
            status={item.hit_top1 === 1 ? "success" : "error"} 
            text={item.hit_top1 === 1 ? "Success" : "Failed"}
          />
        </div>
        {!imageError ? (
          <div style={{ position: 'relative', display: 'inline-block' }}>
            <Image
              src={imagePath}
              alt={`Screenshot for ${fileName}`}
              style={{ maxWidth: '100%', maxHeight: '60vh' }}
              onError={() => setImageError(true)}
              preview={false}
              placeholder={
                <div style={{ 
                  height: 300, 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'center',
                  background: '#f5f5f5'
                }}>
                  Loading...
                </div>
              }
            />
            
            {/* 注意力图叠加层 */}
            {displayOptions.showAttention && attn_score && (() => {
              const heatmapUrl = generateAttentionHeatmap(
                attn_score, 
                img_size[0], 
                img_size[1], 
                n_width_n_height[0], 
                n_width_n_height[1]
              );
              
              return heatmapUrl ? (
                <img
                  src={heatmapUrl}
                  alt="Attention heatmap"
                  style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    height: '100%',
                    pointerEvents: 'none',
                    mixBlendMode: 'multiply'
                  }}
                />
              ) : null;
            })()}
            
            {/* 原本的红色边界框 */}
            {displayOptions.showBbox && (
              <div style={bboxStyle}></div>
            )}
            
            {/* hit_top1坐标点 */}
            {displayOptions.showHitTop1 && (() => {
              const pointStyle = getTop1Point();
              return pointStyle ? <div style={pointStyle}></div> : null;
            })()}
            
            {/* hit_topk所有坐标点 */}
            {displayOptions.showHitTopk && getCoordinatePoints().map((style, index) => (
              <div key={`topk-point-${index}`} style={style}></div>
            ))}
          </div>
        ) : (
          <Alert
            message="Image not found"
            description={fileName}
            type="warning"
            showIcon
            style={{ height: 200 }}
          />
        )}
      </div>
    );
  };

  if (loading) return <div>Loading evaluation data...</div>;
  if (error) return <div>Error: {error}</div>;
  if (mainData.length === 0) return <div>No main data available</div>;

  return (
    <div style={{ padding: '24px', background: '#f0f2f5', minHeight: '100vh' }}>
      <div style={{ maxWidth: '1400px', margin: '0 auto' }}>
        {/* Header */}
        <Card style={{ marginBottom: '24px' }}>
          <Row align="middle" justify="space-between">
            <Col>
              <Title level={2} style={{ margin: 0 }}>
                <EyeOutlined style={{ marginRight: '12px', color: '#1890ff' }} />
                Dual Evaluation Results Viewer
              </Title>
            </Col>
            <Col>
              <Space size="large">
                <Statistic 
                  title="Total Results" 
                  value={statistics.total} 
                  valueStyle={{ color: '#1890ff' }}
                />
                <Statistic 
                  title="Success Rate" 
                  value={statistics.successRate} 
                  suffix="%" 
                  valueStyle={{ color: statistics.successRate > 50 ? '#52c41a' : '#ff4d4f' }}
                />
                <Statistic 
                  title="Avg Overlap" 
                  value={statistics.avgOverlap} 
                  valueStyle={{ color: '#722ed1' }}
                />
              </Space>
            </Col>
          </Row>
        </Card>

        {/* Filters */}
        <Card title="Filters (Applied to Main Data)" style={{ marginBottom: '24px' }}>
          <Row gutter={[16, 16]}>
            <Col xs={24} sm={12} md={8} lg={4}>
              <Text strong>UI Type</Text>
              <Select
                placeholder="All UI Types"
                value={filters.ui_type}
                onChange={(value) => handleFilterChange('ui_type', value)}
                style={{ width: '100%', marginTop: '8px' }}
                allowClear
              >
                {filterOptions.ui_types.map(type => (
                  <Option key={type} value={type}>
                    <Tag color={getUITypeColor(type)}>{type}</Tag>
                  </Option>
                ))}
              </Select>
            </Col>

            <Col xs={24} sm={12} md={8} lg={4}>
              <Text strong>Group</Text>
              <Select
                placeholder="All Groups"
                value={filters.group}
                onChange={(value) => handleFilterChange('group', value)}
                style={{ width: '100%', marginTop: '8px' }}
                allowClear
              >
                {filterOptions.groups.map(group => (
                  <Option key={group} value={group}>{group}</Option>
                ))}
              </Select>
            </Col>

            <Col xs={24} sm={12} md={8} lg={4}>
              <Text strong>Platform</Text>
              <Select
                placeholder="All Platforms"
                value={filters.platform}
                onChange={(value) => handleFilterChange('platform', value)}
                style={{ width: '100%', marginTop: '8px' }}
                allowClear
              >
                {filterOptions.platforms.map(platform => (
                  <Option key={platform} value={platform}>
                    <Tag color={getPlatformColor(platform)}>{platform}</Tag>
                  </Option>
                ))}
              </Select>
            </Col>

            <Col xs={24} sm={12} md={8} lg={4}>
              <Text strong>Application</Text>
              <Select
                placeholder="All Applications"
                value={filters.application}
                onChange={(value) => handleFilterChange('application', value)}
                style={{ width: '100%', marginTop: '8px' }}
                allowClear
              >
                {filterOptions.applications.map(app => (
                  <Option key={app} value={app}>{app}</Option>
                ))}
              </Select>
            </Col>

            <Col xs={24} sm={12} md={8} lg={4}>
              <Text strong>Result Status</Text>
              <Select
                placeholder="All Results"
                value={filters.hit_success}
                onChange={(value) => handleFilterChange('hit_success', value)}
                style={{ width: '100%', marginTop: '8px' }}
                allowClear
              >
                <Option value="success">
                  <CheckCircleOutlined style={{ color: '#52c41a' }} /> Success
                </Option>
                <Option value="failed">
                  <CloseCircleOutlined style={{ color: '#ff4d4f' }} /> Failed
                </Option>
              </Select>
            </Col>
          </Row>
        </Card>

        {/* Display Options */}
        <Card title="Display Options" style={{ marginBottom: '24px' }}>
          <Row gutter={[16, 16]}>
            <Col xs={24} sm={6}>
              <Space>
                <Switch
                  checked={displayOptions.showBbox}
                  onChange={(checked) => setDisplayOptions(prev => ({...prev, showBbox: checked}))}
                />
                <Text>Show BBox (Red Box)</Text>
              </Space>
            </Col>
            <Col xs={24} sm={6}>
              <Space>
                <Switch
                  checked={displayOptions.showAttention}
                  onChange={(checked) => setDisplayOptions(prev => ({...prev, showAttention: checked}))}
                />
                <Text>Show Attention Map</Text>
              </Space>
            </Col>
            <Col xs={24} sm={6}>
              <Space>
                <Switch
                  checked={displayOptions.showHitTop1}
                  onChange={(checked) => setDisplayOptions(prev => ({...prev, showHitTop1: checked}))}
                />
                <Text>Show Hit Top1 Point</Text>
              </Space>
            </Col>
            <Col xs={24} sm={6}>
              <Space>
                <Switch
                  checked={displayOptions.showHitTopk}
                  onChange={(checked) => setDisplayOptions(prev => ({...prev, showHitTopk: checked}))}
                />
                <Text>Show Hit TopK Points</Text>
              </Space>
            </Col>
          </Row>
        </Card>

        {/* Main Content */}
        {filteredData.length > 0 && currentMainItem ? (
          <Row gutter={24}>
            {/* Image Section */}
            <Col xs={24} lg={12}>
              <Card 
                title={
                  <Space>
                    <EyeOutlined />
                    Screenshots Preview
                  </Space>
                }
                style={{ height: 'fit-content' }}
              >
                {/* 主数据图片 */}
                <ImageWithBbox 
                  item={currentMainItem} 
                  isCompare={false}
                  title="Main Data"
                />
                
                {/* 对比数据图片 */}
                <Divider />
                {currentCompareItem ? (
                  <ImageWithBbox 
                    item={currentCompareItem} 
                    isCompare={true}
                    title="Compare Data"
                  />
                ) : (
                  <Alert
                    message="No matching compare data found"
                    description="No corresponding -beta.png file found in compare dataset"
                    type="info"
                    showIcon
                  />
                )}
              </Card>
            </Col>

            {/* Details Section */}
            <Col xs={24} lg={12}>
              <Card title={<><InfoCircleOutlined /> Details</>}>
                <Space direction="vertical" style={{ width: '100%' }} size="large">
                  {/* Main Data Status and Metrics */}
                  <div>
                    <Text strong style={{ fontSize: '16px', color: '#1890ff' }}>Main Data Results</Text>
                    <Row gutter={16} style={{ marginTop: '12px' }}>
                      <Col span={12}>
                        <Card size="small" style={{ textAlign: 'center' }}>
                          <Statistic
                            title="Hit Status"
                            value={currentMainItem.hit_top1 === 1 ? "Success" : "Failed"}
                            valueStyle={{ 
                              color: currentMainItem.hit_top1 === 1 ? '#52c41a' : '#ff4d4f',
                              fontSize: '18px'
                            }}
                            prefix={currentMainItem.hit_top1 === 1 ? 
                              <CheckCircleOutlined /> : <CloseCircleOutlined />}
                          />
                        </Card>
                      </Col>
                      <Col span={12}>
                        <Card size="small" style={{ textAlign: 'center' }}>
                          <Statistic
                            title="Overlap Score"
                            value={currentMainItem.overlap_top1}
                            precision={3}
                            valueStyle={{ color: '#722ed1', fontSize: '18px' }}
                          />
                        </Card>
                      </Col>
                    </Row>
                  </div>

                  {/* Compare Data Status */}
                  {currentCompareItem && (
                    <div>
                      <Text strong style={{ fontSize: '16px', color: '#52c41a' }}>Compare Data Results</Text>
                      <Row gutter={16} style={{ marginTop: '12px' }}>
                        <Col span={12}>
                          <Card size="small" style={{ textAlign: 'center' }}>
                            <Statistic
                              title="Hit Status"
                              value={currentCompareItem.hit_top1 === 1 ? "Success" : "Failed"}
                              valueStyle={{ 
                                color: currentCompareItem.hit_top1 === 1 ? '#52c41a' : '#ff4d4f',
                                fontSize: '18px'
                              }}
                              prefix={currentCompareItem.hit_top1 === 1 ? 
                                <CheckCircleOutlined /> : <CloseCircleOutlined />}
                            />
                          </Card>
                        </Col>
                        <Col span={12}>
                          <Card size="small" style={{ textAlign: 'center' }}>
                            <Statistic
                              title="Overlap Score"
                              value={currentCompareItem.overlap_top1}
                              precision={3}
                              valueStyle={{ color: '#722ed1', fontSize: '18px' }}
                            />
                          </Card>
                        </Col>
                      </Row>
                    </div>
                  )}

                  {/* File Information */}
                  <div>
                    <Text strong>File Name:</Text>
                    <div style={{ 
                      background: '#f5f5f5', 
                      padding: '8px 12px', 
                      borderRadius: '6px',
                      marginTop: '8px',
                      wordBreak: 'break-all'
                    }}>
                      <Text code>{currentMainItem.file_name}</Text>
                    </div>
                  </div>

                  {/* Instruction */}
                  <div>
                    <Text strong>Instruction:</Text>
                    <Card size="small" style={{ marginTop: '8px' }}>
                      <Paragraph style={{ margin: 0 }}>
                        {currentMainItem.instruction}
                      </Paragraph>
                    </Card>
                  </div>

                  {/* Coordinates Comparison */}
                  <div>
                    <Text strong>Coordinates Comparison:</Text>
                    <div style={{ marginTop: '8px' }}>
                      <div style={{ marginBottom: '8px' }}>
                        <Text strong style={{ fontSize: '14px' }}>Main Data:</Text>
                        <div style={{ display: 'flex', gap: '12px', marginTop: '4px' }}>
                          <div>
                            <Text type="secondary" style={{ fontSize: '12px' }}>BBox:</Text>
                            <div>
                              <Tag color="geekblue">
                                [{currentMainItem.bbox_x1y1x2y2.map(coord => coord.toFixed(3)).join(', ')}]
                              </Tag>
                            </div>
                          </div>
                          {currentMainItem.hit_topk_cord && currentMainItem.hit_topk_cord[0] && (
                            <div>
                              <Text type="secondary" style={{ fontSize: '12px' }}>Pred:</Text>
                              <div>
                                <Tag color="green">
                                  [{currentMainItem.hit_topk_cord[0].map(coord => coord.toFixed(3)).join(', ')}]
                                </Tag>
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                      
                      {currentCompareItem && (
                        <div>
                          <Text strong style={{ fontSize: '14px' }}>Compare Data:</Text>
                          <div style={{ display: 'flex', gap: '12px', marginTop: '4px' }}>
                            <div>
                              <Text type="secondary" style={{ fontSize: '12px' }}>BBox:</Text>
                              <div>
                                <Tag color="geekblue">
                                  [{currentCompareItem.bbox_x1y1x2y2.map(coord => coord.toFixed(3)).join(', ')}]
                                </Tag>
                              </div>
                            </div>
                            {currentCompareItem.hit_topk_cord && currentCompareItem.hit_topk_cord[0] && (
                              <div>
                                <Text type="secondary" style={{ fontSize: '12px' }}>Pred:</Text>
                                <div>
                                  <Tag color="green">
                                    [{currentCompareItem.hit_topk_cord[0].map(coord => coord.toFixed(3)).join(', ')}]
                                  </Tag>
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Metadata Tags */}
                  <div>
                    <Text strong>Metadata:</Text>
                    <div style={{ marginTop: '8px' }}>
                      <Space wrap>
                        <Tag color={getPlatformColor(currentMainItem.platform)}>
                          Platform: {currentMainItem.platform}
                        </Tag>
                        <Tag color="blue">
                          App: {currentMainItem.application}
                        </Tag>
                        <Tag color={getUITypeColor(currentMainItem.ui_type)}>
                          UI: {currentMainItem.ui_type}
                        </Tag>
                        <Tag color="purple">
                          Group: {currentMainItem.group}
                        </Tag>
                      </Space>
                    </div>
                  </div>
                </Space>
              </Card>
            </Col>
          </Row>
        ) : (
          <Card style={{ textAlign: 'center', padding: '60px 20px' }}>
            <Title level={3} type="secondary">No results found</Title>
            <Text type="secondary">
              Try adjusting your filter criteria to see more results.
            </Text>
          </Card>
        )}

        {/* Pagination */}
        {filteredData.length > 0 && (
          <Card style={{ marginTop: '24px' }}>
            <Row align="middle" justify="space-between">
              <Col>
                <Space>
                  <Button 
                    type="primary"
                    icon={<LeftOutlined />}
                    onClick={goToPrevious}
                    disabled={currentIndex === 0}
                  >
                    Previous
                  </Button>
                  <Button 
                    type="primary"
                    onClick={goToNext}
                    disabled={currentIndex === filteredData.length - 1}
                  >
                    Next
                    <RightOutlined />
                  </Button>
                </Space>
              </Col>
              
              <Col>
                <Space align="center">
                  <Text strong>
                    {currentIndex + 1} of {filteredData.length}
                  </Text>
                  <Text>Go to:</Text>
                  <InputNumber
                    min={1}
                    max={filteredData.length}
                    value={currentIndex + 1}
                    onChange={goToPage}
                    style={{ width: '80px' }}
                  />
                </Space>
              </Col>
            </Row>
          </Card>
        )}
      </div>
    </div>
  );
};

export default EvaluationViewer;


// import React, { useState, useEffect, useMemo } from 'react';
// import { 
//   Card, 
//   Select, 
//   Row, 
//   Col, 
//   Button, 
//   Tag, 
//   Statistic, 
//   Space, 
//   Typography, 
//   Divider,
//   Alert,
//   Image,
//   Badge,
//   InputNumber,
//   Tooltip,
//   Switch
// } from 'antd';
// import { 
//   LeftOutlined, 
//   RightOutlined, 
//   CheckCircleOutlined, 
//   CloseCircleOutlined,
//   EyeOutlined,
//   InfoCircleOutlined
// } from '@ant-design/icons';

// const { Title, Text, Paragraph } = Typography;
// const { Option } = Select;

// const EVAL_FILE_PATH = '/evaluation_results2.json';
// const getImagePath = (fileName) => `/images2/${fileName}`;


// // 模拟数据 - 你需要替换为实际的 JSON 数据加载
// // const mockData = [
// //   {
// //     "file_name": "android_studio_mac/screenshot_2024-11-28_15-16-55-beta.png",
// //     "ui_type": "icon",
// //     "group": "Dev",
// //     "platform": "macos",
// //     "application": "android_studio",
// //     "id": "android_studio_macos_0",
// //     "instruction": "modify the highlights of the photo with in the virtual android machine in android studio",
// //     "img_size": [455, 351],
// //     "bbox_x1y1x2y2": [0.16483516483516483, 0.33903133903133903, 0.9098901098901099, 0.4301994301994302],
// //     "hit_top1": 1,
// //     "overlap_top1": 0.8,
// //     "overlap_topk": 0.9
// //   },
// //   {
// //     "file_name": "android_studio_mac/screenshot_2024-11-28_15-16-56-beta.png",
// //     "ui_type": "text",
// //     "group": "UI",
// //     "platform": "windows",
// //     "application": "vscode",
// //     "id": "vscode_windows_1",
// //     "instruction": "click the save button in the toolbar to save the current file",
// //     "img_size": [600, 400],
// //     "bbox_x1y1x2y2": [0.1, 0.2, 0.3, 0.4],
// //     "hit_top1": 0,
// //     "overlap_top1": 0.2,
// //     "overlap_topk": 0.3
// //   },
// //   {
// //     "file_name": "vscode_linux/screenshot_2024-11-29_10-30-15-beta.png",
// //     "ui_type": "text",
// //     "group": "Editor",
// //     "platform": "linux",
// //     "application": "sublime_text",
// //     "id": "sublime_linux_2",
// //     "instruction": "find and replace text in the current document",
// //     "img_size": [800, 600],
// //     "bbox_x1y1x2y2": [0.2, 0.3, 0.8, 0.7],
// //     "hit_top1": 1,
// //     "overlap_top1": 0.95,
// //     "overlap_topk": 0.98
// //   }
// // ];

// const EvaluationViewer = () => {
//   const [data, setData] = useState([]);
//   const [loading, setLoading] = useState(true);
//   const [error, setError] = useState(null);

//   const [filters, setFilters] = useState({
//     ui_type: undefined,
//     group: undefined,
//     platform: undefined,
//     application: undefined,
//     hit_success: undefined
//   });
//   const [currentIndex, setCurrentIndex] = useState(0);
//   const [displayOptions, setDisplayOptions] = useState({
//     showAttention: false,
//     showBbox: true,        // 控制原本的红色边界框
//     showHitTop1: false,    // 控制hit_topk_cord第一个坐标点
//     showHitTopk: false     // 控制hit_topk_cord所有坐标点
//   });

//   // 数据加载
//   useEffect(() => {
//     loadEvaluationData();
//   }, []);

//   // 获取所有唯一值用于筛选选项
//   const filterOptions = useMemo(() => {
//     return {
//       ui_types: [...new Set(data.map(item => item.ui_type))].filter(Boolean),
//       groups: [...new Set(data.map(item => item.group))].filter(Boolean),
//       platforms: [...new Set(data.map(item => item.platform))].filter(Boolean),
//       applications: [...new Set(data.map(item => item.application))].filter(Boolean)
//     };
//   }, [data]);

//   // 根据筛选条件过滤数据
//   const filteredData = useMemo(() => {
//     return data.filter(item => {
//       if (filters.ui_type && item.ui_type !== filters.ui_type) return false;
//       if (filters.group && item.group !== filters.group) return false;
//       if (filters.platform && item.platform !== filters.platform) return false;
//       if (filters.application && item.application !== filters.application) return false;
//       if (filters.hit_success === 'success' && item.hit_top1 !== 1) return false;
//       if (filters.hit_success === 'failed' && item.hit_top1 !== 0) return false;
//       return true;
//     });
//   }, [data, filters]);

//   // 统计数据
//   const statistics = useMemo(() => {
//     const total = filteredData.length;
//     const successful = filteredData.filter(item => item.hit_top1 === 1).length;
//     const successRate = total > 0 ? ((successful / total) * 100).toFixed(1) : 0;
//     const avgOverlap = total > 0 ? 
//       (filteredData.reduce((sum, item) => sum + item.overlap_top1, 0) / total).toFixed(3) : 0;
    
//     return { total, successful, successRate, avgOverlap };
//   }, [filteredData]);

//   // 重置当前索引当筛选条件改变时
//   useEffect(() => {
//     setCurrentIndex(0);
//   }, [filters]);

//   const currentItem = filteredData[currentIndex];

//   const handleFilterChange = (key, value) => {
//     setFilters(prev => ({
//       ...prev,
//       [key]: value
//     }));
//   };

//   const goToNext = () => {
//     if (currentIndex < filteredData.length - 1) {
//       setCurrentIndex(currentIndex + 1);
//     }
//   };

//   const goToPrevious = () => {
//     if (currentIndex > 0) {
//       setCurrentIndex(currentIndex - 1);
//     }
//   };

//   const goToPage = (pageIndex) => {
//     const index = pageIndex - 1;
//     if (index >= 0 && index < filteredData.length) {
//       setCurrentIndex(index);
//     }
//   };

//   const getPlatformColor = (platform) => {
//     const colors = {
//       'macos': 'blue',
//       'windows': 'green',
//       'linux': 'orange',
//       'android': 'purple',
//       'ios': 'cyan'
//     };
//     return colors[platform] || 'default';
//   };

//   const getUITypeColor = (uiType) => {
//     const colors = {
//       'icon': 'magenta',
//       'button': 'blue',
//       'text': 'green',
//       'input': 'orange',
//       'menu': 'purple'
//     };
//     return colors[uiType] || 'default';
//   };

//   const loadEvaluationData = async () => {
//     try {
//       setLoading(true);
//       const response = await fetch(EVAL_FILE_PATH);
//       if (!response.ok) {
//         throw new Error('Failed to load evaluation data');
//       }
//       const jsonData = await response.json();
//       setData(jsonData);
//     } catch (err) {
//       setError(err.message);
//     } finally {
//       setLoading(false);
//     }
//   };
//   const generateAttentionHeatmap = (attnScores, imageWidth, imageHeight, nWidth, nHeight) => {
//     if (!attnScores || !Array.isArray(attnScores) || attnScores.length === 0) return null;
    
//     try {
//       // 将注意力分数重塑为二维数组 (对应Python中的reshape)
//       const scores = attnScores[0]; // 取第一个元素，对应Python中的attn_scores[0]
//       const scoresArray = [];
      
//       for (let i = 0; i < nHeight; i++) {
//         const row = [];
//         for (let j = 0; j < nWidth; j++) {
//           row.push(scores[i * nWidth + j]);
//         }
//         scoresArray.push(row);
//       }
      
//       // 找到最小值和最大值进行归一化
//       let minScore = Infinity;
//       let maxScore = -Infinity;
//       for (let i = 0; i < nHeight; i++) {
//         for (let j = 0; j < nWidth; j++) {
//           minScore = Math.min(minScore, scoresArray[i][j]);
//           maxScore = Math.max(maxScore, scoresArray[i][j]);
//         }
//       }
      
//       // 创建归一化的分数映射
//       const normalizedScores = [];
//       for (let i = 0; i < nHeight; i++) {
//         const row = [];
//         for (let j = 0; j < nWidth; j++) {
//           const normalized = (scoresArray[i][j] - minScore) / (maxScore - minScore);
//           row.push(Math.floor(normalized * 255));
//         }
//         normalizedScores.push(row);
//       }
      
//       // 创建canvas并绘制热力图
//       const canvas = document.createElement('canvas');
//       canvas.width = imageWidth;
//       canvas.height = imageHeight;
//       const ctx = canvas.getContext('2d');
      
//       // 创建临时canvas用于缩放
//       const tempCanvas = document.createElement('canvas');
//       tempCanvas.width = nWidth;
//       tempCanvas.height = nHeight;
//       const tempCtx = tempCanvas.getContext('2d');
//       const tempImageData = tempCtx.createImageData(nWidth, nHeight);
      
//       // 应用jet colormap（简化版）
//       const applyJetColormap = (value) => {
//         const normalized = value / 255.0;
//         let r, g, b;
        
//         if (normalized < 0.25) {
//           r = 0;
//           g = Math.floor(normalized * 4 * 255);
//           b = 255;
//         } else if (normalized < 0.5) {
//           r = 0;
//           g = 255;
//           b = Math.floor(255 - (normalized - 0.25) * 4 * 255);
//         } else if (normalized < 0.75) {
//           r = Math.floor((normalized - 0.5) * 4 * 255);
//           g = 255;
//           b = 0;
//         } else {
//           r = 255;
//           g = Math.floor(255 - (normalized - 0.75) * 4 * 255);
//           b = 0;
//         }
        
//         return [r, g, b, Math.floor(0.6 * 255)]; // 0.6透明度对应blend alpha=0.3的效果
//       };
      
//       // 填充临时canvas
//       for (let i = 0; i < nHeight; i++) {
//         for (let j = 0; j < nWidth; j++) {
//           const pixelIndex = (i * nWidth + j) * 4;
//           const [r, g, b, a] = applyJetColormap(normalizedScores[i][j]);
//           tempImageData.data[pixelIndex] = r;
//           tempImageData.data[pixelIndex + 1] = g;
//           tempImageData.data[pixelIndex + 2] = b;
//           tempImageData.data[pixelIndex + 3] = a;
//         }
//       }
      
//       tempCtx.putImageData(tempImageData, 0, 0);
      
//       // 缩放到目标尺寸 (对应Python中的resize，使用NEAREST)
//       ctx.imageSmoothingEnabled = false; // 对应NEAREST重采样
//       ctx.drawImage(tempCanvas, 0, 0, imageWidth, imageHeight);
      
//       return canvas.toDataURL();
//     } catch (error) {
//       console.error('Error generating attention heatmap:', error);
//       return null;
//     }
//   };

//   const ImageWithBbox = ({ item }) => {
//     const [imageError, setImageError] = useState(false);

//     if (!item) return null;

//     const { bbox_x1y1x2y2, img_size, file_name, attn_score, hit_topk_cord, n_width_n_height } = item;
//     const [x1, y1, x2, y2] = bbox_x1y1x2y2;
//     const imagePath = getImagePath(file_name);

//     // 红色边界框样式
//     const bboxStyle = {
//       position: 'absolute',
//       left: `${x1 * 100}%`,
//       top: `${y1 * 100}%`,
//       width: `${(x2 - x1) * 100}%`,
//       height: `${(y2 - y1) * 100}%`,
//       border: '1px solid #ff4d4f',
//       backgroundColor: 'rgba(255, 77, 79, 0.1)',
//       pointerEvents: 'none',
//       borderRadius: '1px'
//     };

//     // 生成坐标点样式
//     const getCoordinatePoints = () => {
//       if (!hit_topk_cord || !Array.isArray(hit_topk_cord)) return [];
      
//       const colors = ['#52c41a', '#1890ff', '#722ed1', '#eb2f96', '#fa8c16'];
      
//       return hit_topk_cord.map((coords, index) => {
//         if (!coords || coords.length < 2) return null;
        
//         // 假设坐标格式为[x, y]或[x1, y1, x2, y2]，取中心点
//         let centerX, centerY;
//         if (coords.length === 2) {
//           [centerX, centerY] = coords;
//         } else if (coords.length === 4) {
//           const [x1, y1, x2, y2] = coords;
//           centerX = (x1 + x2) / 2;
//           centerY = (y1 + y2) / 2;
//         } else {
//           return null;
//         }

//         return {
//           position: 'absolute',
//           left: `${centerX * 100}%`,
//           top: `${centerY * 100}%`,
//           width: '4px',
//           height: '4px',
//           borderRadius: '50%',
//           backgroundColor: colors[index % colors.length],
//           border: '1px solid white',
//           transform: 'translate(-50%, -50%)', // 居中对齐
//           pointerEvents: 'none',
//           zIndex: 10,
//           boxShadow: '0 2px 4px rgba(0,0,0,0.3)'
//         };
//       }).filter(Boolean);
//     };

//     // 获取第一个坐标点（hit_top1）
//     const getTop1Point = () => {
//       if (!hit_topk_cord || !Array.isArray(hit_topk_cord) || hit_topk_cord.length === 0) return null;
      
//       const coords = hit_topk_cord[0];
//       if (!coords || coords.length < 2) return null;
      
//       let centerX, centerY;
//       if (coords.length === 2) {
//         [centerX, centerY] = coords;
//       } else if (coords.length === 4) {
//         const [x1, y1, x2, y2] = coords;
//         centerX = (x1 + x2) / 2;
//         centerY = (y1 + y2) / 2;
//       } else {
//         return null;
//       }

//       return {
//         position: 'absolute',
//         left: `${centerX * 100}%`,
//         top: `${centerY * 100}%`,
//         width: '5px',
//         height: '5px',
//         borderRadius: '50%',
//         backgroundColor: '#52c41a',
//         border: '2px solid white',
//         transform: 'translate(-50%, -50%)',
//         pointerEvents: 'none',
//         zIndex: 11,
//         boxShadow: '0 3px 6px rgba(0,0,0,0.4)'
//       };
//     };

//     return (
//       <div className="relative">
//         {!imageError ? (
//           <div style={{ position: 'relative', display: 'inline-block' }}>
//             <Image
//               src={imagePath}
//               alt={`Screenshot for ${file_name}`}
//               style={{ maxWidth: '100%', maxHeight: '60vh' }}
//               onError={() => setImageError(true)}
//               preview={false}
//               placeholder={
//                 <div style={{ 
//                   height: 300, 
//                   display: 'flex', 
//                   alignItems: 'center', 
//                   justifyContent: 'center',
//                   background: '#f5f5f5'
//                 }}>
//                   Loading...
//                 </div>
//               }
//             />
            
//             {/* 注意力图叠加层 */}
//             {displayOptions.showAttention && attn_score && (() => {
//               // const n_width = 32;
//               // const n_height = 32;
//               const heatmapUrl = generateAttentionHeatmap(
//                 attn_score, 
//                 img_size[0], 
//                 img_size[1], 
//                 n_width_n_height[0], 
//                 n_width_n_height[1]
//               );
              
//               return heatmapUrl ? (
//                 <img
//                   src={heatmapUrl}
//                   alt="Attention heatmap"
//                   style={{
//                     position: 'absolute',
//                     top: 0,
//                     left: 0,
//                     width: '100%',
//                     height: '100%',
//                     pointerEvents: 'none',
//                     mixBlendMode: 'multiply'
//                   }}
//                 />
//               ) : null;
//             })()}
            
//             {/* 原本的红色边界框 */}
//             {displayOptions.showBbox && (
//               <div style={bboxStyle}></div>
//             )}
            
//             {/* hit_top1坐标点 */}
//             {displayOptions.showHitTop1 && (() => {
//               const pointStyle = getTop1Point();
//               return pointStyle ? <div style={pointStyle}></div> : null;
//             })()}
            
//             {/* hit_topk所有坐标点 */}
//             {displayOptions.showHitTopk && getCoordinatePoints().map((style, index) => (
//               <div key={`topk-point-${index}`} style={style}></div>
//             ))}
//           </div>
//         ) : (
//           <Alert
//             message="Image not found"
//             description={file_name}
//             type="warning"
//             showIcon
//             style={{ height: 200 }}
//           />
//         )}
//       </div>
//     );
//   };

//   if (loading) return <div>Loading evaluation data...</div>;
//   if (error) return <div>Error: {error}</div>;
//   if (data.length === 0) return <div>No data available</div>;

//   return (
//     <div style={{ padding: '24px', background: '#f0f2f5', minHeight: '100vh' }}>
//       <div style={{ maxWidth: '1400px', margin: '0 auto' }}>
//         {/* Header */}
//         <Card style={{ marginBottom: '24px' }}>
//           <Row align="middle" justify="space-between">
//             <Col>
//               <Title level={2} style={{ margin: 0 }}>
//                 <EyeOutlined style={{ marginRight: '12px', color: '#1890ff' }} />
//                 Evaluation Results Viewer
//               </Title>
//             </Col>
//             <Col>
//               <Space size="large">
//                 <Statistic 
//                   title="Total Results" 
//                   value={statistics.total} 
//                   valueStyle={{ color: '#1890ff' }}
//                 />
//                 <Statistic 
//                   title="Success Rate" 
//                   value={statistics.successRate} 
//                   suffix="%" 
//                   valueStyle={{ color: statistics.successRate > 50 ? '#52c41a' : '#ff4d4f' }}
//                 />
//                 <Statistic 
//                   title="Avg Overlap" 
//                   value={statistics.avgOverlap} 
//                   valueStyle={{ color: '#722ed1' }}
//                 />
//               </Space>
//             </Col>
//           </Row>
//         </Card>

//         {/* Filters */}
//         <Card title="Filters" style={{ marginBottom: '24px' }}>
//           <Row gutter={[16, 16]}>
//             <Col xs={24} sm={12} md={8} lg={4}>
//               <Text strong>UI Type</Text>
//               <Select
//                 placeholder="All UI Types"
//                 value={filters.ui_type}
//                 onChange={(value) => handleFilterChange('ui_type', value)}
//                 style={{ width: '100%', marginTop: '8px' }}
//                 allowClear
//               >
//                 {filterOptions.ui_types.map(type => (
//                   <Option key={type} value={type}>
//                     <Tag color={getUITypeColor(type)}>{type}</Tag>
//                   </Option>
//                 ))}
//               </Select>
//             </Col>

//             <Col xs={24} sm={12} md={8} lg={4}>
//               <Text strong>Group</Text>
//               <Select
//                 placeholder="All Groups"
//                 value={filters.group}
//                 onChange={(value) => handleFilterChange('group', value)}
//                 style={{ width: '100%', marginTop: '8px' }}
//                 allowClear
//               >
//                 {filterOptions.groups.map(group => (
//                   <Option key={group} value={group}>{group}</Option>
//                 ))}
//               </Select>
//             </Col>

//             <Col xs={24} sm={12} md={8} lg={4}>
//               <Text strong>Platform</Text>
//               <Select
//                 placeholder="All Platforms"
//                 value={filters.platform}
//                 onChange={(value) => handleFilterChange('platform', value)}
//                 style={{ width: '100%', marginTop: '8px' }}
//                 allowClear
//               >
//                 {filterOptions.platforms.map(platform => (
//                   <Option key={platform} value={platform}>
//                     <Tag color={getPlatformColor(platform)}>{platform}</Tag>
//                   </Option>
//                 ))}
//               </Select>
//             </Col>

//             <Col xs={24} sm={12} md={8} lg={4}>
//               <Text strong>Application</Text>
//               <Select
//                 placeholder="All Applications"
//                 value={filters.application}
//                 onChange={(value) => handleFilterChange('application', value)}
//                 style={{ width: '100%', marginTop: '8px' }}
//                 allowClear
//               >
//                 {filterOptions.applications.map(app => (
//                   <Option key={app} value={app}>{app}</Option>
//                 ))}
//               </Select>
//             </Col>

//             <Col xs={24} sm={12} md={8} lg={4}>
//               <Text strong>Result Status</Text>
//               <Select
//                 placeholder="All Results"
//                 value={filters.hit_success}
//                 onChange={(value) => handleFilterChange('hit_success', value)}
//                 style={{ width: '100%', marginTop: '8px' }}
//                 allowClear
//               >
//                 <Option value="success">
//                   <CheckCircleOutlined style={{ color: '#52c41a' }} /> Success
//                 </Option>
//                 <Option value="failed">
//                   <CloseCircleOutlined style={{ color: '#ff4d4f' }} /> Failed
//                 </Option>
//               </Select>
//             </Col>
//           </Row>
//         </Card>
//         {/* Display Options */}
//         <Card title="Display Options" style={{ marginBottom: '24px' }}>
//           <Row gutter={[16, 16]}>
//             <Col xs={24} sm={6}>
//               <Space>
//                 <Switch
//                   checked={displayOptions.showBbox}
//                   onChange={(checked) => setDisplayOptions(prev => ({...prev, showBbox: checked}))}
//                 />
//                 <Text>Show BBox (Red Box)</Text>
//               </Space>
//             </Col>
//             <Col xs={24} sm={6}>
//               <Space>
//                 <Switch
//                   checked={displayOptions.showAttention}
//                   onChange={(checked) => setDisplayOptions(prev => ({...prev, showAttention: checked}))}
//                 />
//                 <Text>Show Attention Map</Text>
//               </Space>
//             </Col>
//             <Col xs={24} sm={6}>
//               <Space>
//                 <Switch
//                   checked={displayOptions.showHitTop1}
//                   onChange={(checked) => setDisplayOptions(prev => ({...prev, showHitTop1: checked}))}
//                 />
//                 <Text>Show Hit Top1 Point</Text>
//               </Space>
//             </Col>
//             <Col xs={24} sm={6}>
//               <Space>
//                 <Switch
//                   checked={displayOptions.showHitTopk}
//                   onChange={(checked) => setDisplayOptions(prev => ({...prev, showHitTopk: checked}))}
//                 />
//                 <Text>Show Hit TopK Points</Text>
//               </Space>
//             </Col>
//           </Row>
//         </Card>

//         {/* Main Content */}
//         {filteredData.length > 0 && currentItem ? (
//           <Row gutter={24}>
//             {/* Image Section */}
//             <Col xs={24} lg={12}>
//               <Card 
//                 title={
//                   <Space>
//                     <EyeOutlined />
//                     Screenshot Preview
//                     <Badge 
//                       status={currentItem.hit_top1 === 1 ? "success" : "error"} 
//                       text={currentItem.hit_top1 === 1 ? "Success" : "Failed"}
//                     />
//                   </Space>
//                 }
//                 style={{ height: 'fit-content' }}
//               >
//                 <ImageWithBbox item={currentItem} />
//               </Card>
//             </Col>

//             {/* Details Section */}
//             <Col xs={24} lg={12}>
//               <Card title={<><InfoCircleOutlined /> Details</>}>
//                 <Space direction="vertical" style={{ width: '100%' }} size="large">
//                   {/* Status and Metrics */}
//                   <div>
//                     <Row gutter={16}>
//                       <Col span={12}>
//                         <Card size="small" style={{ textAlign: 'center' }}>
//                           <Statistic
//                             title="Hit Status"
//                             value={currentItem.hit_top1 === 1 ? "Success" : "Failed"}
//                             valueStyle={{ 
//                               color: currentItem.hit_top1 === 1 ? '#52c41a' : '#ff4d4f',
//                               fontSize: '18px'
//                             }}
//                             prefix={currentItem.hit_top1 === 1 ? 
//                               <CheckCircleOutlined /> : <CloseCircleOutlined />}
//                           />
//                         </Card>
//                       </Col>
//                       <Col span={12}>
//                         <Card size="small" style={{ textAlign: 'center' }}>
//                           <Statistic
//                             title="Overlap Score"
//                             value={currentItem.overlap_top1}
//                             precision={3}
//                             valueStyle={{ color: '#722ed1', fontSize: '18px' }}
//                           />
//                         </Card>
//                       </Col>
//                     </Row>
//                   </div>

//                   {/* File Information */}
//                   <div>
//                     <Text strong>File Name:</Text>
//                     <div style={{ 
//                       background: '#f5f5f5', 
//                       padding: '8px 12px', 
//                       borderRadius: '6px',
//                       marginTop: '8px',
//                       wordBreak: 'break-all'
//                     }}>
//                       <Text code>{currentItem.file_name}</Text>
//                     </div>
//                   </div>

//                   {/* Instruction */}
//                   <div>
//                     <Text strong>Instruction:</Text>
//                     <Card size="small" style={{ marginTop: '8px' }}>
//                       <Paragraph style={{ margin: 0 }}>
//                         {currentItem.instruction}
//                       </Paragraph>
//                     </Card>
//                   </div>

//                   {/* BBox Coordinates */}
//                   {/* <div>
//                     <Text strong>
//                       BBox Coordinates 
//                       <Tooltip title="Normalized coordinates [x1, y1, x2, y2]">
//                         <InfoCircleOutlined style={{ marginLeft: '4px', color: '#1890ff' }} />
//                       </Tooltip>
//                     </Text>
//                     <div style={{ marginTop: '8px' }}>
//                       <Tag color="geekblue">
//                         [{currentItem.bbox_x1y1x2y2.map(coord => coord.toFixed(3)).join(', ')}]
//                       </Tag>
//                     </div>
//                   </div> */}
//                   <div style={{ display: 'flex', gap: '16px' }}>
//                     {/* 第一个 div */}
//                     <div>
//                       <Text strong>
//                         BBox Coordinates 
//                         <Tooltip title="Normalized coordinates [x1, y1, x2, y2]">
//                           <InfoCircleOutlined style={{ marginLeft: '4px', color: '#1890ff' }} />
//                         </Tooltip>
//                       </Text>
//                       <div style={{ marginTop: '8px' }}>
//                         <Tag color="geekblue">
//                           [{currentItem.bbox_x1y1x2y2.map(coord => coord.toFixed(3)).join(', ')}]
//                         </Tag>
//                       </div>
//                     </div>
                    
//                     {/* 第二个 div */}
//                     <div>
//                       <Text strong>
//                         Pred Coordinates
//                         <Tooltip title="Predicted coordinates [x1, y1, x2, y2]">
//                           <InfoCircleOutlined style={{ marginLeft: '4px', color: '#1890ff' }} />
//                         </Tooltip>
//                       </Text>
//                       <div style={{ marginTop: '8px' }}>
//                         <Tag color="green">
//                           [{currentItem.hit_topk_cord[0].map(coord => coord.toFixed(3)).join(', ')}]
//                         </Tag>
//                       </div>
//                     </div>
//                   </div>
                  

//                   {/* Metadata Tags */}
//                   <div>
//                     <Text strong>Metadata:</Text>
//                     <div style={{ marginTop: '8px' }}>
//                       <Space wrap>
//                         <Tag color={getPlatformColor(currentItem.platform)}>
//                           Platform: {currentItem.platform}
//                         </Tag>
//                         <Tag color="blue">
//                           App: {currentItem.application}
//                         </Tag>
//                         <Tag color={getUITypeColor(currentItem.ui_type)}>
//                           UI: {currentItem.ui_type}
//                         </Tag>
//                         <Tag color="purple">
//                           Group: {currentItem.group}
//                         </Tag>
//                       </Space>
//                     </div>
//                   </div>
//                 </Space>
//               </Card>
//             </Col>
//           </Row>
//         ) : (
//           <Card style={{ textAlign: 'center', padding: '60px 20px' }}>
//             <Title level={3} type="secondary">No results found</Title>
//             <Text type="secondary">
//               Try adjusting your filter criteria to see more results.
//             </Text>
//           </Card>
//         )}

//         {/* Pagination */}
//         {filteredData.length > 0 && (
//           <Card style={{ marginTop: '24px' }}>
//             <Row align="middle" justify="space-between">
//               <Col>
//                 <Space>
//                   <Button 
//                     type="primary"
//                     icon={<LeftOutlined />}
//                     onClick={goToPrevious}
//                     disabled={currentIndex === 0}
//                   >
//                     Previous
//                   </Button>
//                   <Button 
//                     type="primary"
//                     onClick={goToNext}
//                     disabled={currentIndex === filteredData.length - 1}
//                   >
//                     Next
//                     <RightOutlined />
//                   </Button>
//                 </Space>
//               </Col>
              
//               <Col>
//                 <Space align="center">
//                   <Text strong>
//                     {currentIndex + 1} of {filteredData.length}
//                   </Text>
//                   <Text>Go to:</Text>
//                   <InputNumber
//                     min={1}
//                     max={filteredData.length}
//                     value={currentIndex + 1}
//                     onChange={goToPage}
//                     style={{ width: '80px' }}
//                   />
//                 </Space>
//               </Col>
//             </Row>
//           </Card>
//         )}
//       </div>
//     </div>
//   );
// };

// export default EvaluationViewer;
