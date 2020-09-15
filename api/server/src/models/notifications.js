'use strict';
module.exports = (sequelize, DataTypes) => {
  const notifications = sequelize.define('notifications', {
    id: {
      type: DataTypes.UUID,
      primaryKey: true,
      defaultValue: DataTypes.UUIDV1,
    },
    projectId: {
      type: DataTypes.UUID,
      allowNull: true,
      unique: false,
    },
    userId: {
      type: DataTypes.UUID,
      allowNull: true,
      unique: false,
    },
    content: {
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
      allowNull: true,
      unique: false,
      defaultValue: "Unseen"
    }
  }, {});
  notifications.associate = function(models) {
    // associations can be defined here
  };
  return notifications;
};