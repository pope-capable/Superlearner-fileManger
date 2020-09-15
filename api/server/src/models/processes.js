'use strict';
module.exports = (sequelize, DataTypes) => {
  const processes = sequelize.define('processes', {
    id: {
      type: DataTypes.UUID,
      primaryKey: true,
      defaultValue: DataTypes.UUIDV1,
    },
    projectId: {
      type: DataTypes.UUID,
      allowNull: false,
      unique: false,
    },
    name: {
      type: DataTypes.STRING,
      allowNull: false,
      unique: false,
    },
    type: {
      type: DataTypes.STRING,
      allowNull: false,
      unique: false
    },
    status: {
      type: DataTypes.STRING,
      allowNull: false,
      unique: false,
      defaultValue: "Running..."
    },
    result: {
      type: DataTypes.UUID,
      allowNull: true,
      unique: false,
    },
    failure_reason: {
      type: DataTypes.STRING,
      allowNull: true,
      unique: false
    }
  }, {});
  processes.associate = function(models) {
    processes.belongsTo (models.files_mains, {foreignKey: 'result'})
    // associations can be defined here
  };
  return processes;
};